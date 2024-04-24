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

#ifdef KINECT_AZURE
// include Kinect libraries
#endif


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
   * The acquired frame is stored in the #_k4a_rgbd, #_rgbd and #_rgb 
   * attributes.
   *
   * @see set_params
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type acquire_frame(bool dummy = false) {
    // acquire last frame from the camera device
    // if camera device is a Kinect Azure, use the Azure SDK
    // and translate the frame in OpenCV format
    #ifdef KINECT_AZURE
    // acquire and translate into _rgb and _rgbd
    #else
    // acquire and store into _rgb (RGB) and _rgbd (RGBD), if available
    #endif
    return return_type::success;
  }
  

  /* LEFT BRANCH =============================================================*/

  /**
   * @brief Compute the skeleton from the depth map.
   *
   * Compute the skeleton from the depth map. The resulting skeleton is stored
   * in #_skeleton3D attribute as a map of 3D points.
   *
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type skeleton_from_depth_compute(bool debug = false) {
  #ifdef KINECT_AZURE
    return return_type::success;
  #else
    // NOOP
    return return_type::success;
  #endif 
  }

  /**
   * @brief Remove unnecessary points from the point cloud
   *
   * Make the point cloud lighter by removing unnecessary points, so that it 
   * can be sent to the database via network
   * 
   * @author Nicola
   * @return result status ad defined in return_type
   */
  return_type point_cloud_filter(bool debug = false) {
    #ifdef KINECT_AZURE
    return return_type::success;
    #else
    // NOOP
    return return_type::success;
    #endif
  }

  /**
   * @brief Transform the 3D skeleton coordinates in the global reference frame
   * 
   * Use the extrinsic camera parameters to transorm the 3D skeleton coordinates
   * just before sending them as plugin output.
   * 
   * @return return_type 
   */
  return_type coordinate_transfrom(bool debug = false) {
    return return_type::success;
  }


  /* RIGHT BRANCH ============================================================*/

  /**
   * @brief Compute the skeleton from RGB images only
   *
   * Compute the skeleton from RGB images only. On success, the field 
   * #_skeleton2D is updated (as a map of 2D points).
   * Also, the field #_heatmaps is updated with the joints heatmaps (one per 
   * joint).
   * 
   * There is a configuration flag for optionally skipping this branch 
   * on Azure agents.
   * 
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type skeleton_from_rgb_compute(bool debug = false) {
    return return_type::success;
  }

  /**
   * @brief Compute the hessians for joints
   *
   * Compute the hessians for joints on the RGB frame based on the #_heatmaps 
   * field.
   * 
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type hessian_compute(bool debug = false) {
    return return_type::success;
  }

  /**
   * @brief Compute the 3D covariance matrix
   * 
   * Compute the 3D covariance matrix.
   * Two possible cases:
   *   1. one Azure camera: use the 3D to uncertainty in the view axis, use 
   *      the 2D image to uncertainty in the projection plane
   *   2. one RGB camera: calculates a 3D ellipsoid based on the 2D covariance
   *      plus the "reasonable" depth range as a third azis (direction of view)
   *
   * @author Alessandro
   * @return result status ad defined in return_type
   */
  return_type cov3D_compute(bool debug = false) {
    return return_type::success;
  }

  /**
   * @brief Consistency check of the 3D skeleton according to human physiology
   *
   * @authors Marco, Matteo
   * @return result status ad defined in return_type
   */
  return_type consistency_check(bool debug = false) {
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
  return_type get_output(json *out, vector<unsigned_char> *blob) override {
    // call in sequence the methods to compute the skeleton (acquire_frame, 
    // skeleton_from_depth_compute, etc.)
    acquire_frame(json["debug"]["acquire_frame"]);
    skeleton_from_depth_compute();
    skeleton_from_rgb_compute();
    hessian_compute();
    cov3D_compute();
    consistency_check();
    point_cloud_filter();
    coordinate_transfrom();
    // store the output in the out parameter json and the point cloud in the 
    // blob parameter
  }

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
  vector<Mat> _heatmaps;           /**< the joints heatmaps */
  Mat _point_cloud;                /**< the filtered body point cloud */
  Mat _cov2D;                      /**< the 2D covariance matrix (18x3)*/
  Mat _cov3D;                      /**< the 3D covariance matrix */
  Mat _cov3D_adj;                  /**< the adjusted 3D covariance matrix */
  json _params;                    /**< the parameters of the plugin */
  #ifdef KINECT_AZURE
  k4a_capture_t _k4a_rgbd;         /**< the last capture */
  #endif
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
