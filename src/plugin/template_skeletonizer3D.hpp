/*
  ____  _        _      _              _             _____ ____
 / ___|| | _____| | ___| |_ ___  _ __ (_)_______ _ _|___ /|  _ \
 \___ \| |/ / _ \ |/ _ \ __/ _ \| '_ \| |_  / _ \ '__||_ \| | | |
  ___) |   <  __/ |  __/ || (_) | | | | |/ /  __/ |  ___) | |_| |
 |____/|_|\_\___|_|\___|\__\___/|_| |_|_/___\___|_| |____/|____/
 */

#include "../source.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <pugg/Kernel.h>

using namespace cv;
using namespace std;
using json = nlohmann::json;

/**
 * @class Skeletonizer3D
 *
 * @brief Skeletonizer3D is a plugin that computes the 3D skeleton of a human
 * body from a depth map.
 *
 */
class Skeletonizer3D : public Source<json> {
public:
  /**
   * @brief Constructor
   *
   */
  Skeletonizer3D();

  /**
   * @brief Destructor
   *
   * @author Paolo
   */
  ~Skeletonizer3D();

  /**
   * @brief Acquire a frame from a camera device. Camera ID is defined in the
   * parameters list.
   *
   * The acquired frame is stored in the #_rgbd and #_rgb attributes.
   *
   * @see set_params
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type acquire_frame() {
    return return_type::success;
  }
  
  /**
   * @brief Compute the skeleton from the depth map.
   *
   * Compute the skeleton from the depth map. The resulting skeleton is stored
   * in _skeleton3D attribute as a map of 3D points.
   *
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type skeleton_from_depth_compute() {
    return return_type::success;
  }

  /**
   * @brief Compute the skeleton from RGB images only
   *
   * Compute the skeleton from RGB images only
   * 
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type skeleton_from_rgb_compute() {
    return return_type::success;
  }

  /**
   * @brief Compute the joints heatmap
   *
   * Compute the joints heatmap
   * 
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type point_cloud_filter() {
    return return_type::success;
  }

  /**
   * @brief Compute the point cloud
   *
   * Compute the point cloud
   * 
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type hessian_compute() {
    return return_type::success;
  }

  /**
   * @brief Compute the 2D covariance matrix
   * 
   * Compute the 2D covariance matrix
   *
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type cov3D_from_cov2D_compute() {
    return return_type::success;
  }

  /**
   * @brief cov3D_adjust
   *
   * @authors Marco, Matteo
   * @return result status ad defined in return_type
   */
  return_type cov3D_adjust() {
    return return_type::success;
  }

  /*
    ____  _             _             _          __  __
   |  _ \| |_   _  __ _(_)_ __    ___| |_ _   _ / _|/ _|
   | |_) | | | | |/ _` | | '_ \  / __| __| | | | |_| |_
   |  __/| | |_| | (_| | | | | | \__ \ |_| |_| |  _|  _|
   |_|   |_|\__,_|\__, |_|_| |_| |___/\__|\__,_|_| |_|
                  |___/
  */

  /**
   * @brief Set the parameters of the plugin
   * 
   * The parameters are stored in the #_params attribute. This method shall be
   * called imediately after the plugin is instantiated
   *
   * @author Paolo
   * @param params
   */
  void set_params(void *params) override { _params = *(json *)params; }

  /**
   * @brief Get the output of the plugin
   * 
   * This method acquires a new image and computes the skeleton from it.
   *
   * @author Paolo
   * @param out The output of the plugin as JSON
   * @param blob Possible additional binary data
   * @return return_type
   */
  return_type get_output(json *out, vector<unsigned_char> *blob) override {}

  /**
   * @brief Provide further info to Miroscic agent
   * 
   * Provide the Miroscic agent loading this plugin with further info to be
   * printed after initialization
   *
   * @return a map with the information of the plugin
   */
  map<string, string> info() override {}

  /**
   * @brief The plugin identifier
   *
   * @author Paolo
   * @return a string with plugin kind
   */
  string kind() override {}

protected:
  Mat _rgbd;                       /**< the last RGBD frame */
  Mat _rgb;                        /**< the last RGB frame */
  map<string, vector> _skeleton2D; /**< the skeleton from 2D cameras only*/
  map<string, vector> _skeleton3D; /**< the skeleton from 3D cameras only*/
  Mat _heatmap;                    /**< the joints heatmap */
  Mat _point_cloud;                /**< the body point cloud */
  Mat _cov2D;                      /**< the 2D covariance matrix */
  Mat _cov3D;                      /**< the 3D covariance matrix */
  Mat _cov3D_adj;                  /**< the adjusted 3D covariance matrix */
  json _params;                    /**< the parameters of the plugin */
}

INSTALL_SOURCE_DRIVER(Skeletonizer3D, json);

/*
Example of JSON parameters:
{
    "device": id or relative_video_uri,
    "resolution_rgbd": "lxh",
    "resolution_rgb": "lxh",
    "fps": 30,
    "extrinsic": [[diagonal],[off_diagonal],[translation]]
}
*/