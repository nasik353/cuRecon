#include "app/hdf5_handler.h"
#include "cuRecon/quaternion.h"
#include <algorithm>
#include <fmt/core.h>
#include <regex>
#include <stdexcept>

namespace hdf5
{

    HDF5ImageLoader::HDF5ImageLoader(const std::string &file_path) : file_id(-1)
    {

        file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (file_id < 0)
        {
            throw HDF5Exception("Failed to open HDF5 file: " + file_path);
        }
    }

    HDF5ImageLoader::~HDF5ImageLoader()
    {
        if (file_id >= 0)
        {
            H5Fclose(file_id);
        }
    }

    image HDF5ImageLoader::load_image(const std::string &frame_name)
    {
        std::string dataset_path = fmt::format("/skin/{}/image", frame_name);

        hid_t dataset_id = H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT);
        if (dataset_id < 0)
        {
            throw HDF5Exception("Failed to open dataset: " + dataset_path);
        }

        hid_t dataspace_id = H5Dget_space(dataset_id);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);

        std::vector<unsigned char> raw_data(dims[0] * dims[1]);
        image image_data = {dims[0], dims[1], raw_data};

        H5Dread(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, image_data.raw_data.data());

        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);

        return image_data;
    }

    pointcloud HDF5ImageLoader::load_pointcloud(const std::string &frame_name)
    {
        std::string dataset_path = fmt::format("/skin/{}/processed/pointcloud", frame_name);

        hid_t dataset_id = H5Dopen2(file_id, dataset_path.c_str(), H5P_DEFAULT);
        if (dataset_id < 0)
        {
            throw HDF5Exception("Failed to open dataset: " + dataset_path);
        }

        hid_t dataspace_id = H5Dget_space(dataset_id);
        hsize_t dims[2];
        H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);

        if (dims[1] != 9)
        {
            H5Sclose(dataspace_id);
            H5Dclose(dataset_id);
            throw HDF5Exception("Pointcloud data does not have 9 columns.");
        }

        std::vector<float> raw_data(dims[0] * dims[1]);
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw_data.data());

        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);

        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float minZ = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float maxY = std::numeric_limits<float>::lowest();
        float maxZ = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < dims[0]; ++i)
        {
            float x = raw_data[i * 9];
            float y = raw_data[i * 9 + 1];
            float z = raw_data[i * 9 + 2];

            if (x < minX)
                minX = x;
            if (y < minY)
                minY = y;
            if (z < minZ)
                minZ = z;
            if (x > maxX)
                maxX = x;
            if (y > maxY)
                maxY = y;
            if (z > maxZ)
                maxZ = z;
        }

        float centerX = (minX + maxX) / 2.0f;
        float centerY = (minY + maxY) / 2.0f;
        float centerZ = (minZ + maxZ) / 2.0f;
        float scale = 2.0f / std::max({maxX - minX, maxY - minY, maxZ - minZ});

        pointcloud cloud;
        for (size_t i = 0; i < dims[0]; ++i)
        {
            float x = raw_data[i * 9];
            float y = raw_data[i * 9 + 1];
            float z = raw_data[i * 9 + 2];

            float nx = (x - centerX) * scale;
            float ny = (y - centerY) * scale;
            float nz = (z - centerZ) * scale;

            float r = raw_data[i * 9 + 3] / 255.0f;
            float g = raw_data[i * 9 + 4] / 255.0f;
            float b = raw_data[i * 9 + 5] / 255.0f;

            cloud.data.push_back({nx, ny, nz, r, g, b});
        }
        return cloud;
    }

    frame HDF5ImageLoader::load_frame(const std::string &frame_name)
    {
        frame f;
        f._id = std::stoi(frame_name.substr(frame_name.find('_') + 1));
        f._q_init = quaternion_from_axis_angle(0, 0, -1, 5 * f._id);
        f._q_trans = quaternion_from_axis_angle(0, 0, -1, 5 * f._id);
        f._image = load_image(frame_name);
        f._pointcloud = rotate_pointcloud(load_pointcloud(frame_name), f._q_init);
        return f;
    }

    std::vector<std::string> HDF5ImageLoader::get_frame_names() const
    {
        hid_t skin_group_id = H5Gopen(file_id, "/skin", H5P_DEFAULT);
        if (skin_group_id < 0)
        {
            throw HDF5Exception("Failed to open group: /skin");
        }
        std::vector<std::string> frame_names = get_object_names(skin_group_id);
        H5Gclose(skin_group_id);
        return frame_names;
    }

    hid_t HDF5ImageLoader::create_or_open_group(const std::string &group_path)
    {
        if (H5Lexists(file_id, group_path.c_str(), H5P_DEFAULT) > 0)
        {
            return H5Gopen(file_id, group_path.c_str(), H5P_DEFAULT);
        }
        else
        {
            return H5Gcreate2(file_id, group_path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }
    }

    std::vector<std::string> HDF5ImageLoader::get_object_names(hid_t loc_id) const
    {
        std::vector<std::string> object_names;
        hsize_t num_objs;
        H5Gget_num_objs(loc_id, &num_objs);
        for (hsize_t i = 0; i < num_objs; i++)
        {
            char name[1024];
            H5Gget_objname_by_idx(loc_id, i, name, sizeof(name));
            object_names.push_back(std::string(name));
        }

        std::sort(object_names.begin(), object_names.end(), [](const std::string &a, const std::string &b)
                  {
        std::regex frame_regex("frame_(\\d+)");
        std::smatch match_a, match_b;

        if (std::regex_search(a, match_a, frame_regex) && std::regex_search(b, match_b, frame_regex)) {
            return std::stoi(match_a[1]) < std::stoi(match_b[1]);
        }
        return a < b; });

        return object_names;
    }

} // namespace hdf5
