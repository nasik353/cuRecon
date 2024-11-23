#ifndef HDF5_HANDLER_H
#define HDF5_HANDLER_H

#include <hdf5.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "cuRecon/types.h"

namespace hdf5
{

  class HDF5Exception : public std::runtime_error
  {
  public:
    explicit HDF5Exception(const std::string &message) : std::runtime_error(message) {}
  };

  class HDF5ImageLoader
  {
  public:
    explicit HDF5ImageLoader(const std::string &file_path);
    ~HDF5ImageLoader();

    HDF5ImageLoader(const HDF5ImageLoader &) = delete;
    HDF5ImageLoader &operator=(const HDF5ImageLoader &) = delete;

    image load_image(const std::string &frame_name);
    pointcloud load_pointcloud(const std::string &frame_name);
    frame load_frame(const std::string &frame_name);
    std::vector<std::string> get_frame_names() const;

  private:
    hid_t file_id;

    hid_t create_or_open_group(const std::string &group_path);
    std::vector<std::string> get_object_names(hid_t loc_id) const;
  };

} // namespace hdf5

#endif // HDF5_HANDLER_H