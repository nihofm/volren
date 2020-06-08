#include "volume.h"
#include <vector>
#include <iostream>
#include <fstream>

#if defined(WITH_OPENVDB)
#include <openvdb/openvdb.h>
#endif

Volume::Volume() : model(1), absorbtion_coefficient(0.1), scattering_coefficient(0.5), phase_g(0) {}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, float density) : Volume() {
    std::vector<float> data(w * h * d, density);
    texture = Texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, data.data(), true);
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data) : Volume() {
    texture = Texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data, true);
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data) : Volume() {
    texture = Texture3D(name, w, h, d, GL_R16, GL_RED, GL_UNSIGNED_SHORT, data, true);
}

Volume::Volume(const std::string& name, size_t w, size_t h, size_t d, const float* data) : Volume() {
    texture = Texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, data, true);
}

Volume::Volume(const fs::path& path) : Volume() {
    const fs::path extension = path.extension();
    if (extension == ".dat") {
        // TODO handle .dat
    }
    else if (extension == ".raw") { // handle .raw
        std::ifstream raw(path, std::ios::binary);
        if (raw.is_open()) {
            // assumes file name of structure: name_WxHxD_type.raw
            size_t last = 0, next = 0, w = 0, h = 0, d = 0;
            std::string name, filename = path.filename();
            // prase name
            if ((next = filename.find("_", last)) != std::string::npos) {
                name = filename.substr(last, next-last);
                last = next + 1;
            }
            // parse WxHxD
            if ((next = filename.find("_", last)) != std::string::npos) {
                if ((next = filename.find("x", last)) != std::string::npos) {
                    w = std::stoi(filename.substr(last, next-last));
                    last = next + 1;
                }
                if ((next = filename.find("x", last)) != std::string::npos) {
                    h = std::stoi(filename.substr(last, next-last));
                    last = next + 1;
                }
                if ((next = filename.find("_", last)) != std::string::npos) {
                    d = std::stoi(filename.substr(last, next-last));
                    last = next + 1;
                }
                last = next + 1;
            }
            // parse data type and setup volume texture
            std::vector<uint8_t> data(std::istreambuf_iterator<char>(raw), {});
            if (filename.find("uint8"))
                texture = Texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), true);
            else if (filename.find("uint16"))
                texture = Texture3D(name, w, h, d, GL_R16, GL_RED, GL_UNSIGNED_SHORT, (uint16_t*)data.data(), true);
            else if (filename.find("float"))
                texture = Texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, (float*)data.data(), true);
            else {
                std::cerr << "WARN: Unable to parse data type from raw file name: " << path << " -> falling back to uint8_t." << std::endl;
                texture = Texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), true);
            }
        } else
            throw std::runtime_error("Volume: Unable to read file: " + path.string());
    }
#if defined(WITH_OPENVDB)
    else if (extension == ".vdb") { // handle .vdb TODO FIXME some .vdb crash in Texture3D GL upload
        // open file
        openvdb::initialize();
        openvdb::io::File vdb_file(path.string());
        vdb_file.open();
        // load grid
        openvdb::GridBase::Ptr baseGrid = 0;
        for (openvdb::io::File::NameIterator nameIter = vdb_file.beginName();
                nameIter != vdb_file.endName(); ++nameIter) {
            if (nameIter.gridName() == "density")
                baseGrid = vdb_file.readGrid(nameIter.gridName());
        }
        vdb_file.close();
        // cast to FloatGrid
        if (!baseGrid) throw std::runtime_error("Volume: No OpenVDB density grid found in " + path.string());
        const openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        const openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
        const openvdb::Coord dim = grid->evalActiveVoxelDim() + openvdb::Coord(1); // inclusive bounds
        float min_value, max_value;
        grid->evalMinMax(min_value, max_value);
        // read into linearized array of uint8_t
        std::vector<uint8_t> data(dim.x() * dim.y() * dim.z());
        for (auto iter = grid->cbeginValueOn(); iter.test(); ++iter) {
            if (iter.isVoxelValue()) {
                const float value = *iter;
                const auto coord = iter.getCoord() - box.getStart();
                data[coord.z() * dim.x() * dim.y() + coord.y() * dim.x() + coord.x()] = uint8_t(std::round(value * 255.f));
            }
        }
        // load into GL texture
        texture = Texture3D(path.stem(), dim.x(), dim.y(), dim.z(), GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), true);
    }
#endif
    else
        throw std::runtime_error("Volume: Unable to load file extension: " + extension.string());
}

Volume::~Volume() {

}
