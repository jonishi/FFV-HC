// Moving STL demo: initialization, motion/distribution, block search and output.
#include "Solver.h"
#include "BlockBoundingBox.h"
#include "Polylib.h"
#include "FFVGlobalVars.h"
#include "FFVPM.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace PolylibNS;

void Solver::InitSTL_movingstl() {
	if (this->pl_movingstl) {
		return;
	}

	pl_movingstl_initial = new BCMPolylib;
	pl_movingstl_initial->load(g_pFFVConfig->PolylibConfig);

	pl_movingstl = new BCMPolylib;
	pl_movingstl->load(g_pFFVConfig->PolylibConfig);

	if (myrank == 0) {
		const Vec3r rootOrigin = g_pFFVConfig->RootBlockOrigin;
		const double rootLength = g_pFFVConfig->RootBlockLength;
		const int margin = g_pFFVConfig->LeafBlockNumberOfVirtualCells;
		BlockBoundingBox bbb(tree, rootOrigin, rootLength, size, margin);

		std::vector<Node*>& nodes = tree->getLeafNodeArray();
		for (int rank = 0; rank < MPI::COMM_WORLD.Get_size(); ++rank) {
			BoundingBox box;
			for (int id = partition->getStart(rank); id < partition->getEnd(rank); ++id) {
				box.addBox(bbb.getBoundingBox(nodes[id]));
			}
			this->pl_movingstl->set_bounding_box(rank, box);
		}
		this->pl_movingstl->send_to_all();
	} else {
		this->pl_movingstl->load_from_rank0();
	}
}

void Solver::UpdateSTL(int step) {
	MPI_Barrier(MPI_COMM_WORLD);
	PM_Start(tm_UpdateSTL);
	const double start = MPI_Wtime();

	MoveSTL(step);

	const double searchStart = MPI_Wtime();
	SearchSTL();
	times[9] += MPI_Wtime() - searchStart;

	OutputSTL(step);

	const double elapsed[5] = {MPI_Wtime() - start, times[8], times[9], times[10], times[11]};
	PM_Stop(tm_UpdateSTL);
	MPI_Allreduce(elapsed, &times[7], 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void Solver::MoveSTL(int step) {
	const double moveStart = MPI_Wtime();

	const Vec3d center = g_pFFVConfig->STLRotationCenter;
	const double radius = g_pFFVConfig->STLRotationRadius;
	const double omega = g_pFFVConfig->STLRotationAngularVelocity;
	const double time = step * g_pFFVConfig->TimeControlTimeStepDeltaT;

	if (myrank == 0) {
		const double cosTheta = std::cos(omega*time);
		const double sinTheta = std::sin(omega*time);
		std::vector<PolygonGroup*>* initial = pl_movingstl_initial->get_leaf_groups();
		std::vector<PolygonGroup*>* current = pl_movingstl->get_leaf_groups();
		for (size_t g = 0; g < initial->size(); ++g) {
			const std::vector<PrivateTriangle*>* triangles = (*initial)[g]->get_triangles();
			const size_t count = triangles ? triangles->size() : 0;
			std::vector<PL_REAL> xyz(count*9);
			std::vector<int> ids(count), boundaryIds(count);
			for (size_t i = 0; i < count; ++i) {
				PrivateTriangle* triangle = (*triangles)[i];
				ids[i] = triangle->get_id();
				boundaryIds[i] = triangle->get_exid();
				PolylibNS::Vertex** vertices = triangle->get_vertex();
				for (int k = 0; k < 3; ++k) {
					const double x = (*vertices[k])[0], y = (*vertices[k])[1], z = (*vertices[k])[2];
					xyz[9*i+3*k  ] = center.x + cosTheta*(x + radius) - sinTheta*y;
					xyz[9*i+3*k+1] = center.y + sinTheta*(x + radius) + cosTheta*y;
					xyz[9*i+3*k+2] = center.z + z;
				}
			}
			(*current)[g]->init(xyz.data(), ids.data(), boundaryIds.data(), 0, 0, 0, static_cast<unsigned int>(count));
		}
		delete initial;
		delete current;
	}

	const double moved = MPI_Wtime();
	times[8] = moved - moveStart;

	PM_Start(tm_Init_DistributeSTL, 0, 0, true);
	POLYLIB_STAT status;
	if (myrank == 0) {
		status = pl_movingstl->send_to_all();
	} else {
		status = pl_movingstl->load_from_rank0();
	}
	if (status != PLSTAT_OK) MPI::COMM_WORLD.Abort(EX_FAILURE);
	PM_Stop(tm_Init_DistributeSTL);

	times[9] = MPI_Wtime() - moved;
}

void Solver::SearchSTL() {
	using namespace PolylibNS;
	// SAT: triangle/AABB intersection, including touching faces/edges.
	// Polylib's AABB search alone can return triangles that miss the block.
	const auto intersects = [](PolylibNS::Vertex** vertices, const Vec3r& lo, const Vec3r& length) {
		double v[3][3], edge[3][3], half[3];
		const double origin[3] = {lo.x, lo.y, lo.z};
		const double extent[3] = {length.x, length.y, length.z};
		for (int a = 0; a < 3; ++a) {
			half[a] = extent[a]*0.5;
			for (int k = 0; k < 3; ++k) v[k][a] = (*vertices[k])[a] - (origin[a] + half[a]);
			for (int k = 0; k < 3; ++k) edge[k][a] = v[(k+1)%3][a] - v[k][a];
		}
		const auto separated = [&](double x, double y, double z) {
			const double r = half[0]*std::fabs(x) + half[1]*std::fabs(y) + half[2]*std::fabs(z);
			double low = v[0][0]*x + v[0][1]*y + v[0][2]*z, high = low;
			for (int k = 1; k < 3; ++k) {
				const double p = v[k][0]*x + v[k][1]*y + v[k][2]*z;
				low = std::min(low, p); high = std::max(high, p);
			}
			const double tolerance = 1.e-12*(r + std::fabs(low) + std::fabs(high));
			return low > r + tolerance || high < -r - tolerance;
		};
		if (separated(1,0,0) || separated(0,1,0) || separated(0,0,1)) return false;
		for (int k = 0; k < 3; ++k)
			if (separated(0,edge[k][2],-edge[k][1]) ||
				separated(-edge[k][2],0,edge[k][0]) ||
				separated(edge[k][1],-edge[k][0],0)) return false;
		return !separated(edge[0][1]*edge[1][2]-edge[0][2]*edge[1][1],
			edge[0][2]*edge[1][0]-edge[0][0]*edge[1][2],
			edge[0][0]*edge[1][1]-edge[0][1]*edge[1][0]);
	};

	// Distribution leaves only local current geometry on every rank.
	std::vector<PolygonGroup*>* groups = pl_movingstl->get_leaf_groups();
	stlTriangles.assign(blockManager.getNumBlock(), std::vector<STLTriangle>());
	for (int b = 0; b < blockManager.getNumBlock(); ++b) {
		BlockBase* block = blockManager.getBlock(b);
		Vec3r lo = block->getOrigin();
		Vec3r length = block->getBlockSize();
		Vec3<PL_REAL> lower, upper;
		for (int a = 0; a < 3; ++a) {
			lower[a] = std::nextafter(static_cast<PL_REAL>(lo[a]), -std::numeric_limits<PL_REAL>::infinity());
			upper[a] = std::nextafter(static_cast<PL_REAL>(lo[a]+length[a]), std::numeric_limits<PL_REAL>::infinity());
		}
		BBox box(lower, upper);
		for (size_t g = 0; g < groups->size(); ++g) {
			if (!(*groups)[g]->get_triangles() || (*groups)[g]->get_triangles()->empty()) continue;
			std::vector<PrivateTriangle*> candidates;
			if ((*groups)[g]->search(&box, false, &candidates) != PLSTAT_OK)
				MPI::COMM_WORLD.Abort(EX_FAILURE);
			for (size_t j = 0; j < candidates.size(); ++j) {
				PolylibNS::Vertex** v = candidates[j]->get_vertex();
				if (!intersects(v, lo, length)) continue;
				STLTriangle triangle;
				triangle.group = g;
				triangle.id = candidates[j]->get_id();
				for (int k = 0; k < 3; ++k)
					for (int a = 0; a < 3; ++a) triangle.xyz[3*k+a] = (*v[k])[a];
				stlTriangles[b].push_back(triangle);
			}
		}
	}
	delete groups;
}

void Solver::OutputSTL(int step) {
	const double stlOutputStart = MPI_Wtime();
	const double time = step * g_pFFVConfig->TimeControlTimeStepDeltaT;
	// Preserve every block membership, including duplicate triangles.
	std::vector<const STLTriangle*> triangles;
	std::vector<int> blockIds;
	for (size_t b = 0; b < stlTriangles.size(); ++b)
		for (size_t j = 0; j < stlTriangles[b].size(); ++j) {
			triangles.push_back(&stlTriangles[b][j]);
			blockIds.push_back(partition->getStart(myrank) + b);
		}
	// Explicit little-endian encoding avoids native struct padding/byte order.
	const uint16_t endianProbe = 1;
	const bool littleEndian = *reinterpret_cast<const unsigned char*>(&endianProbe) == 1;
	std::vector<char> bytes;
	const auto appendScalar = [&](const void* value, size_t width) {
		const char* p = static_cast<const char*>(value);
		if (littleEndian) bytes.insert(bytes.end(), p, p + width);
		else for (size_t k = width; k > 0; --k) bytes.push_back(p[k-1]);
	};
	static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
		"Binary STL requires IEEE Float32");
	static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
		"Binary VTP requires IEEE Float64");
	if (triangles.size() > std::numeric_limits<uint32_t>::max()) MPI::COMM_WORLD.Abort(EX_FAILURE);
	bytes.reserve(84 + triangles.size()*50);
	bytes.resize(80, 0);
	const char stlHeader[] = "FFV moving STL (binary)";
	std::copy(stlHeader, stlHeader + sizeof(stlHeader)-1, bytes.begin());
	const uint32_t triangleCount = static_cast<uint32_t>(triangles.size());
	appendScalar(&triangleCount, sizeof(triangleCount));
	for (size_t i = 0; i < triangles.size(); ++i) {
		const double* p = triangles[i]->xyz;
		const double ax = p[3]-p[0], ay = p[4]-p[1], az = p[5]-p[2];
		const double bx = p[6]-p[0], by = p[7]-p[1], bz = p[8]-p[2];
		double nx = ay*bz-az*by, ny = az*bx-ax*bz, nz = ax*by-ay*bx;
		const double norm = std::hypot(std::hypot(nx, ny), nz);
		if (norm > 0.0) { nx /= norm; ny /= norm; nz /= norm; }
		const float normal[3] = {static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz)};
		for (int k = 0; k < 3; ++k) appendScalar(&normal[k], sizeof(float));
		for (int k = 0; k < 9; ++k) {
			const float coordinate = static_cast<float>(p[k]);
			appendScalar(&coordinate, sizeof(coordinate));
		}
		const uint16_t attribute = 0;
		appendScalar(&attribute, sizeof(attribute));
	}
	std::ostringstream stlFilename;
	stlFilename << "data-stl-" << std::setfill('0') << std::setw(5) << myrank
		<< "-" << std::setw(10) << step << ".stl";
	std::ofstream stl(stlFilename.str().c_str(), std::ios::out | std::ios::binary);
	stl.write(bytes.data(), bytes.size());
	stl.close();
	if (!stl) MPI::COMM_WORLD.Abort(EX_FAILURE);
	const double stlWritten = MPI_Wtime();

	// VTK appended raw data: each array is preceded by a UInt64 byte count.
	// VTP retains Float64 coordinates; STL's standard representation is Float32.
	bytes.clear();
	bytes.reserve(7*sizeof(uint64_t) + triangles.size()*120);
	uint64_t offsets[7];
	const auto beginArray = [&](int array, uint64_t size) {
		offsets[array] = bytes.size();
		appendScalar(&size, sizeof(size));
	};
	for (int field = 0; field < 4; ++field) {
		beginArray(field, triangles.size()*sizeof(int32_t));
		for (size_t i = 0; i < triangles.size(); ++i) {
			const int32_t value = field == 0 ? myrank : field == 1 ? blockIds[i] :
				field == 2 ? triangles[i]->group : triangles[i]->id;
			appendScalar(&value, sizeof(value));
		}
	}
	beginArray(4, triangles.size()*9*sizeof(double));
	for (size_t i = 0; i < triangles.size(); ++i)
		for (int k = 0; k < 9; ++k) appendScalar(&triangles[i]->xyz[k], sizeof(double));
	beginArray(5, triangles.size()*3*sizeof(int64_t));
	for (size_t i = 0; i < triangles.size()*3; ++i) {
		const int64_t point = i;
		appendScalar(&point, sizeof(point));
	}
	beginArray(6, triangles.size()*sizeof(int64_t));
	for (size_t i = 0; i < triangles.size(); ++i) {
		const int64_t end = 3*(i+1);
		appendScalar(&end, sizeof(end));
	}
	std::ostringstream vtpFilename;
	vtpFilename << "data-stl-" << std::setfill('0') << std::setw(5) << myrank
		<< "-" << std::setw(10) << step << ".vtp";
	std::ofstream vtp(vtpFilename.str().c_str(), std::ios::out | std::ios::binary);
	vtp << std::scientific << std::setprecision(17)
		<< "<?xml version=\"1.0\"?>\n"
		<< "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
		<< "<PolyData>\n<FieldData>\n"
		<< "<DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">"
		<< time << "</DataArray>\n</FieldData>\n"
		<< "<Piece NumberOfPoints=\"" << triangles.size()*3
		<< "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
		<< triangles.size() << "\">\n<CellData>\n";
	const char* names[] = {"Rank", "BlockID", "GroupID", "TriangleID"};
	for (int field = 0; field < 4; ++field)
		vtp << "<DataArray type=\"Int32\" Name=\"" << names[field]
			<< "\" format=\"appended\" offset=\"" << offsets[field] << "\"/>\n";
	vtp << "</CellData>\n<Points>\n"
		<< "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
		<< offsets[4] << "\"/>\n</Points>\n<Polys>\n"
		<< "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"appended\" offset=\""
		<< offsets[5] << "\"/>\n"
		<< "<DataArray type=\"Int64\" Name=\"offsets\" format=\"appended\" offset=\""
		<< offsets[6] << "\"/>\n</Polys>\n</Piece>\n</PolyData>\n"
		<< "<AppendedData encoding=\"raw\">\n_";
	vtp.write(bytes.data(), bytes.size());
	vtp << "\n</AppendedData>\n</VTKFile>\n";
	vtp.close();
	if (!vtp) MPI::COMM_WORLD.Abort(EX_FAILURE);
	times[10] = stlWritten - stlOutputStart;
	times[11] = MPI_Wtime() - stlWritten;
}
