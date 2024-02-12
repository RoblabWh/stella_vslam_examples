#ifdef HAVE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#endif
#ifdef HAVE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "stella_vslam/system.h"
#include "stella_vslam/config.h"
#include "stella_vslam/camera/base.h"
#include "stella_vslam/util/yaml.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <numeric>
#include <csignal>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

#ifdef USE_STACK_TRACE_LOGGER
#include <backward.hpp>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

std::shared_ptr<stella_vslam::system> slam;
#ifdef HAVE_PANGOLIN_VIEWER
std::shared_ptr<pangolin_viewer::viewer> viewer;
#endif
#ifdef HAVE_SOCKET_PUBLISHER
std::shared_ptr<socket_publisher::publisher> publisher;
#endif

void sighandler(int signum) {
    (void)signum;
    slam->request_terminate();
#ifdef HAVE_PANGOLIN_VIEWER
    if (viewer)
        viewer->request_terminate();
#endif
#ifdef HAVE_SOCKET_PUBLISHER
    if (publisher)
        publisher->request_terminate();
#endif
}

void mono_tracking(const std::vector<std::string>& video_file_paths,
                   const cv::Mat& mask,
                   const unsigned int frame_skip,
                   const std::vector<unsigned int>& start_times,
                   const bool no_sleep,
                   const bool wait_loop_ba,
                   const double start_timestamp,
                   std::vector<double>& track_times) {
    double timestamp = start_timestamp;
    for (unsigned int i = 0; i < video_file_paths.size(); ++i) {
        std::cout << "processing video \"" << video_file_paths[i] << '"' << std::endl;
        auto video = cv::VideoCapture(video_file_paths[i]);
        if (!video.isOpened()) {
            std::cerr << "unable to open video" << std::endl;
            continue;
        }
        video.set(0, start_times[i]);

        cv::Mat frame;

        unsigned int num_frame = 0;

        bool is_not_end = true;
        while (is_not_end) {
            // wait until the loop BA is finished
            if (wait_loop_ba) {
                while (slam->loop_BA_is_running() || !slam->mapping_module_is_enabled()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }

            is_not_end = video.read(frame);

            const auto tp_1 = std::chrono::steady_clock::now();

            if (!frame.empty() && (num_frame % frame_skip == 0)) {
                // input the current frame and estimate the camera pose
                slam->feed_monocular_frame(frame, timestamp, mask);
            }

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            if (num_frame % frame_skip == 0) {
                track_times.push_back(track_time);
            }

            // wait until the timestamp of the next frame
            if (!no_sleep) {
                const auto wait_time = 1.0 / slam->get_camera()->fps_ - track_time;
                if (0.0 < wait_time) {
                    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<unsigned int>(wait_time * 1e6)));
                }
            }

            timestamp += 1.0 / slam->get_camera()->fps_;
            ++num_frame;

            // check if the termination of slam system is requested or not
            if (slam->terminate_is_requested()) {
                break;
            }
        }
    }
}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    backward::SignalHandling sh;
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto video_file_path = op.add<popl::Value<std::string>>("m", "video", "video file path");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto frame_skip = op.add<popl::Value<unsigned int>>("", "frame-skip", "interval of frame skip", 1);
    auto start_time = op.add<popl::Value<unsigned int>>("s", "start-time", "time to start playing [milli seconds]");
    auto no_sleep = op.add<popl::Switch>("", "no-sleep", "not wait for next frame in real time");
    auto wait_loop_ba = op.add<popl::Switch>("", "wait-loop-ba", "wait until the loop BA is finished");
    auto auto_term = op.add<popl::Switch>("", "auto-term", "automatically terminate the viewer");
    auto log_level = op.add<popl::Value<std::string>>("", "log-level", "log level", "info");
    auto eval_log_dir = op.add<popl::Value<std::string>>("", "eval-log-dir", "store trajectory and tracking times at this path (Specify the directory where it exists.)", "");
    auto map_db_path_in = op.add<popl::Value<std::string>>("i", "map-db-in", "load a map from this path", "");
    auto map_db_path_out = op.add<popl::Value<std::string>>("o", "map-db-out", "store a map database at this path after slam", "");
    auto disable_mapping = op.add<popl::Switch>("", "disable-mapping", "disable mapping");
    auto temporal_mapping = op.add<popl::Switch>("", "temporal-mapping", "enable temporal mapping");
    auto start_timestamp = op.add<popl::Value<double>>("t", "start-timestamp", "timestamp of the start of the video capture");
    auto viewer = op.add<popl::Value<std::string>>("", "viewer", "viewer [pangolin_viewer, socket_publisher, none]");
    auto point_cloud_path = op.add<popl::Value<std::string>>("p", "pc-out", "store point cloud at this path after slam", "");
    auto keyframe_path = op.add<popl::Value<std::string>>("k", "kf-out", "store keyframes in this folder after slam", "");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!op.unknown_options().empty()) {
        for (const auto& unknown_option : op.unknown_options()) {
            std::cerr << "unknown_options: " << unknown_option << std::endl;
        }
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !video_file_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::string> video_file_paths;
    for (size_t i = 0; i < video_file_path->count(); ++i) {
        video_file_paths.push_back(video_file_path->value(i));
    }
    std::vector<unsigned int> start_times(video_file_paths.size(), 0);
    for (size_t i = 0; i < start_time->count(); ++i) {
        start_times[i] = start_time->value(i);
    }

    // viewer
    std::string viewer_string;
    if (viewer->is_set()) {
        viewer_string = viewer->value();
        if (viewer_string != "pangolin_viewer" && viewer_string != "socket_publisher" && viewer_string != "none") {
            std::cerr << "invalid arguments (--viewer)" << std::endl
                      << std::endl
                      << op << std::endl;
            return EXIT_FAILURE;
        }
#ifndef HAVE_PANGOLIN_VIEWER
        if (viewer_string == "pangolin_viewer") {
            std::cerr << "pangolin_viewer not linked" << std::endl
                      << std::endl
                      << op << std::endl;
            return EXIT_FAILURE;
        }
#endif
#ifndef HAVE_SOCKET_PUBLISHER
        if (viewer_string == "socket_publisher") {
            std::cerr << "socket_publisher not linked" << std::endl
                      << std::endl
                      << op << std::endl;
            return EXIT_FAILURE;
        }
#endif
    }
    else {
#ifdef HAVE_PANGOLIN_VIEWER
        viewer_string = "pangolin_viewer";
#elif defined(HAVE_SOCKET_PUBLISHER)
        viewer_string = "socket_publisher";
#endif
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    spdlog::set_level(spdlog::level::from_str(log_level->value()));

    // load configuration
    std::shared_ptr<stella_vslam::config> cfg;
    try {
        cfg = std::make_shared<stella_vslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    // You cannot get timestamps of images with this input format.
    // It is recommended to specify the timestamp when the video recording was started in Unix time.
    // If not specified, the current system time is used instead.
    double timestamp = 0.0;
    if (!start_timestamp->is_set()) {
        std::cerr << "--start-timestamp is not set. using system timestamp." << std::endl;
        if (no_sleep->is_set()) {
            std::cerr << "If --no-sleep is set without --start-timestamp, timestamps may overlap between multiple runs." << std::endl;
        }
        std::chrono::system_clock::time_point start_time_system = std::chrono::system_clock::now();
        timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(start_time_system.time_since_epoch()).count();
    }
    else {
        timestamp = start_timestamp->value();
    }

    // build a slam system
    slam = std::make_shared<stella_vslam::system>(cfg, vocab_file_path->value());
    bool need_initialize = true;
    if (map_db_path_in->is_set()) {
        need_initialize = false;
        const auto path = fs::path(map_db_path_in->value());
        if (path.extension() == ".yaml") {
            YAML::Node node = YAML::LoadFile(path);
            for (const auto& map_path : node["maps"].as<std::vector<std::string>>()) {
                if (!slam->load_map_database(path.parent_path() / map_path)) {
                    return EXIT_FAILURE;
                }
            }
        }
        else {
            if (!slam->load_map_database(path)) {
                return EXIT_FAILURE;
            }
        }
    }
    slam->startup(need_initialize);
    if (disable_mapping->is_set()) {
        slam->disable_mapping_module();
    }
    else if (temporal_mapping->is_set()) {
        slam->enable_temporal_mapping();
        slam->disable_loop_detector();
    }

    // load the mask image
    const cv::Mat mask = mask_img_path->is_set() ? cv::imread(mask_img_path->value(), cv::IMREAD_GRAYSCALE) : cv::Mat();

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef HAVE_PANGOLIN_VIEWER
    if (viewer_string == "pangolin_viewer") {
        viewer = std::make_shared<pangolin_viewer::viewer>(
            stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "PangolinViewer"),
            slam,
            slam->get_frame_publisher(),
            slam->get_map_publisher());
    }
#endif
#ifdef HAVE_SOCKET_PUBLISHER
    if (viewer_string == "socket_publisher") {
        publisher = std::make_shared<socket_publisher::publisher>(
            stella_vslam::util::yaml_optional_ref(cfg->yaml_node_, "SocketPublisher"),
            slam,
            slam->get_frame_publisher(),
            slam->get_map_publisher());
    }
#endif

    // set signal handler for clean shutdown on ctrl-c
    signal(SIGINT, sighandler);

    // run tracking
    std::vector<double> track_times;
    std::unique_ptr<std::thread> thread;

    // run the slam in another thread
    thread = std::make_unique<std::thread>([&]() {
        if (slam->get_camera()->setup_type_ == stella_vslam::camera::setup_type_t::Monocular) {
            mono_tracking(video_file_paths,
                          mask,
                          frame_skip->value(),
                          start_times,
                          no_sleep->is_set(),
                          wait_loop_ba->is_set(),
                          timestamp,
                          track_times);
        }
        else {
            throw std::runtime_error("Invalid setup type: " + slam->get_camera()->get_setup_type_string());
        }

        // wait until the loop BA is finished
        while (slam->loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
        std::cout << "finished processing videos" << std::endl;

        // automatically close the viewer
        if (auto_term->is_set()) {
            if (viewer_string == "pangolin_viewer") {
#ifdef HAVE_PANGOLIN_VIEWER
                viewer->request_terminate();
#endif
            }
            if (viewer_string == "socket_publisher") {
#ifdef HAVE_SOCKET_PUBLISHER
                publisher->request_terminate();
#endif
            }
        }
    });

    // run the viewer in the current thread
    if (viewer_string == "pangolin_viewer") {
#ifdef HAVE_PANGOLIN_VIEWER
        viewer->run();
#endif
    }
    if (viewer_string == "socket_publisher") {
#ifdef HAVE_SOCKET_PUBLISHER
        publisher->run();
#endif
    }

    thread->join();

    // shutdown the slam process
    slam->shutdown();

    if (eval_log_dir->is_set()) {
        const std::string& dir_path = eval_log_dir->value();
        // output the trajectories for evaluation
        slam->save_frame_trajectory(dir_path + "/frame_trajectory.txt", "TUM");
        slam->save_keyframe_trajectory(dir_path + "/keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs(dir_path + "/track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }

    if (!track_times.empty())
    {
        std::sort(track_times.begin(), track_times.end());
        const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
        std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
        std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
    }

    if (point_cloud_path->is_set()) {
        if (!slam->save_point_cloud(point_cloud_path->value())) {
            std::cerr << "Unable to save the point cloud." << std::endl;
        }
    }
    if (keyframe_path->is_set()) {
        if (!slam->save_keyframes(keyframe_path->value())) {
            std::cerr << "Unable to save the keyframes." << std::endl;
        }
    }
    if (map_db_path_out->is_set()) {
        if (!slam->save_map_database(map_db_path_out->value())) {
            std::cerr << "Unable to save the map database." << std::endl;
        }
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif
}
