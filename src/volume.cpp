#include "volume.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <glm/gtc/matrix_transform.hpp>

#if defined(WITH_OPENVDB)
#include <openvdb/openvdb.h>
#endif

#if defined(WITH_DCMTK)
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#endif

VolumeImpl::VolumeImpl(const std::string& name) : name(name), model(1), absorbtion_coefficient(0.1), scattering_coefficient(0.5), phase_g(0), slice_thickness(1.f) {}

VolumeImpl::VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, float density) : VolumeImpl(name) {
    std::vector<float> data(w * h * d, density);
    texture = Texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, data.data(), false);
}

VolumeImpl::VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, const uint8_t* data) : VolumeImpl(name) {
    texture = Texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data, false);
}

VolumeImpl::VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, const uint16_t* data) : VolumeImpl(name) {
    texture = Texture3D(name, w, h, d, GL_R16, GL_RED, GL_UNSIGNED_SHORT, data, false);
}

VolumeImpl::VolumeImpl(const std::string& name, size_t w, size_t h, size_t d, const float* data) : VolumeImpl(name) {
    texture = Texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, data, false);
}

VolumeImpl::VolumeImpl(const std::string& name, const fs::path& path) : VolumeImpl(name) {
    const fs::path extension = path.extension();
    if (extension == ".dat") { // handle .dat
        std::ifstream dat_file(path);
        if (!dat_file.is_open())
            throw std::runtime_error("Unable to read file: " + path.string());
        // read meta data
        fs::path raw_path;
        glm::ivec3 dim;
        std::string format;
        int bits;
        while (!dat_file.eof()) {
            std::string key;
            dat_file >> key;
            if (key == "ObjectFileName:") {
                dat_file >> raw_path;
                printf("Scan raw file: %s\n", raw_path.c_str());
            } else if (key == "Resolution:") {
                dat_file >> dim.x;
                dat_file >> dim.y;
                dat_file >> dim.z;
                printf("Scan resolution: %i, %i, %i\n", dim.x, dim.y, dim.z);
            } else if (key == "SliceThickness:") {
                dat_file >> slice_thickness.x;
                dat_file >> slice_thickness.y;
                dat_file >> slice_thickness.z;
                printf("Scan slice thickness: %f, %f, %f\n", slice_thickness.x, slice_thickness.y, slice_thickness.z);
            } else if (key == "Format:") {
                dat_file >> format;
                printf("Scan format: %s\n", format.c_str());
            } else if (key == "BitsUsed:") {
                dat_file >> bits;
                printf("Scan bits used: %i\n", bits);
            } else
                std::cout << "Skipping key: " << key << "..." << std::endl;
        }
        // read raw data
        raw_path = path.parent_path() / raw_path;
        std::ifstream raw_file(raw_path, std::ios::binary);
        if (!raw_file.is_open())
            throw std::runtime_error("Unable to read file: " + raw_path.string());
        // parse data type and setup volume texture
        std::vector<uint8_t> data(std::istreambuf_iterator<char>(raw_file), {});
        std::cout << "data size bytes: " << data.size() << " / " << dim.x*dim.y*dim.z << std::endl;
        if (true || format.find("UCHAR"))
            texture = Texture3D(raw_path.filename(), dim.x, dim.y, dim.z, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), false);
        else if (format.find("USHORT"))
            texture = Texture3D(raw_path.filename(), dim.x, dim.y, dim.z, GL_R16, GL_RED, GL_UNSIGNED_SHORT, (uint16_t*)data.data(), false);
        else if (format.find("FLOAT"))
            texture = Texture3D(raw_path.filename(), dim.x, dim.y, dim.z, GL_R32F, GL_RED, GL_FLOAT, (float*)data.data(), false);
        else {
            std::cerr << "WARN: Unable to parse data type from dat file: " << path << " -> falling back to uint8_t." << std::endl;
            texture = Texture3D(raw_path.filename(), dim.x, dim.y, dim.z, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), false);
        }
        // from z up to y up
        //std::swap(slice_thickness.y, slice_thickness.z);
        model = glm::rotate(glm::mat4(1), float(1.5 * M_PI), glm::vec3(1, 0, 0));
    }
    else if (extension == ".raw") { // handle .raw
        std::ifstream raw(path, std::ios::binary);
        if (!raw.is_open())
            throw std::runtime_error("Unable to read file: " + path.string());
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
        // parse slice thickness WxHxD
        if ((next = filename.find("_", last)) != std::string::npos) {
            if ((next = filename.find("x", last)) != std::string::npos) {
                slice_thickness.x = std::stof(filename.substr(last, next-last));
                last = next + 1;
            }
            if ((next = filename.find("x", last)) != std::string::npos) {
                slice_thickness.y = std::stof(filename.substr(last, next-last));
                last = next + 1;
            }
            if ((next = filename.find("_", last)) != std::string::npos) {
                slice_thickness.z = std::stof(filename.substr(last, next-last));
                last = next + 1;
            }
            last = next + 1;
        }
        // parse data type and setup volume texture
        std::vector<uint8_t> data(std::istreambuf_iterator<char>(raw), {});
        if (filename.find("uint8"))
            texture = Texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), false);
        else if (filename.find("uint16"))
            texture = Texture3D(name, w, h, d, GL_R16, GL_RED, GL_UNSIGNED_SHORT, (uint16_t*)data.data(), false);
        else if (filename.find("float"))
            texture = Texture3D(name, w, h, d, GL_R32F, GL_RED, GL_FLOAT, (float*)data.data(), false);
        else {
            std::cerr << "WARN: Unable to parse data type from raw file name: " << path << " -> falling back to uint8_t." << std::endl;
            texture = Texture3D(name, w, h, d, GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), false);
        }
    }
#if defined(WITH_OPENVDB)
    else if (extension == ".vdb") { // handle .vdb TODO FIXME some .vdb are broken, possibly stride?
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
        if (!baseGrid) throw std::runtime_error("No OpenVDB density grid found in " + path.string());
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
        texture = Texture3D(path.stem(), dim.x(), dim.y(), dim.z(), GL_R8, GL_RED, GL_UNSIGNED_BYTE, data.data(), false);
    }
#endif
#if defined(WITH_DCMTK)
    else if (extension == ".dcm") { // handle .dcm dicom files TODO
        // TODO read data
        DcmFileFormat fileformat;
        OFCondition status = fileformat.loadFile(path.c_str());
        if (!status.good())
            throw std::runtime_error(std::string("Error: cannot read DICOM file: ") + status.text());
        OFString patientName;
        if (fileformat.getDataset()->findAndGetOFString(DCM_PatientName, patientName).good())
            std::cout << "Patient's Name: " << patientName << std::endl;
        else
            std::cerr << "Error: cannot access Patient's Name!" << std::endl;

        // TODO read image
        DicomImage image(path.c_str());
        if (image.getStatus() != EIS_Normal)
            throw std::runtime_error("Unable to load DICOM image " + path.string() + ": " + DicomImage::getString(image.getStatus()));
        if (image.isMonochrome()) {
            image.setMinMaxWindow();
            image.setNoVoiTransformation();
            std::cout << "DICOM: " << image.getWidth() << "x" << image.getHeight() << "x" << image.getDepth() << ", frames: " << image.getFrameCount() << " / " << image.getNumberOfFrames() << std::endl;
            const DiPixel* pixelData = image.getInterData();
            if (pixelData != NULL) {
                /* do something useful with the pixel data */
            }
        } else
            throw std::runtime_error("non-monochrome DICOM image");
    }
#endif
    else
        throw std::runtime_error("Unable to load file extension: " + extension.string());
    // setup model matrix
    model = glm::scale(model, slice_thickness);                         // scale slices
}

VolumeImpl::~VolumeImpl() {

}
