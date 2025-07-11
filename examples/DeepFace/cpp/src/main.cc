
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/inotify.h>
#include <signal.h>
#include <limits.h>
#include <json/json.h>
#include <cassert>

#include "rknn_api.h"

#include "timer.h"
#include "rknn_app.h"
#include "data_utils.h"
#include "path_utils.h"

#define DYNAMIC_SHAPE_COMPATABLE

#ifdef USE_INOTIFY
#define BUF_LEN (10 * (sizeof(struct inotify_event) + NAME_MAX + 1))
#define DASH_EVENT_MASK (IN_MOVED_TO | IN_CLOSE_WRITE)

bool g_quit = false;

void cleanupSigHandler(int s)
{
    g_quit = true;
	printf("Caught signal %d\n",s);
}
#endif

int load_input_data(rknn_app_context_t* rknn_app_ctx, const char* input_path, rknn_app_buffer* input_buffer){
    printf("\nLOAD INPUTS\n");
    int ret = 0;

    if (input_path == NULL){
        return -1;
    }

    std::vector<std::string> input_paths_split;
    input_paths_split = split(input_path, "#");
    if (input_paths_split.size() == 0){
        return -1;
    }

    if (rknn_app_ctx->n_input != input_paths_split.size()) {
        printf("ERROR: input number is not match, input number is %d, but input path is %s\n", rknn_app_ctx->n_input, input_path);
        return -1;
    }
    unsigned char* temp_data = NULL;
    rknn_tensor_type temp_data_type;
    for (int i=0; i<rknn_app_ctx->n_input; i++){
        if (strstr(input_paths_split[i].c_str(), ".npy")) {
            printf("  input[%d] - NPY: %s\n", i, input_paths_split[i].c_str());
            temp_data = (unsigned char*)load_npy(input_paths_split[i].c_str(), &(rknn_app_ctx->in_attr[i]), &temp_data_type);
        } 
        else {
            printf("  input[%d] - IMG: %s\n", i, input_paths_split[i].c_str());
            LETTER_BOX letter_box;
            temp_data = load_image_and_autoresize(input_paths_split[i].c_str(), &letter_box, &(rknn_app_ctx->in_attr[i]));
            temp_data_type = RKNN_TENSOR_UINT8;
        }
        if (!temp_data) {
            printf("ERROR: load input data failed\n");
            return -1;
        }

        ret = rknn_app_wrap_input_buffer(rknn_app_ctx, temp_data, temp_data_type, &input_buffer[i], i);
        free(temp_data);
    }

    return ret;
}


int loop_run(rknn_app_context_t* rknn_app_ctx, rknn_app_buffer* input_buffer, rknn_app_buffer* output_buffer, int loop_time){
    printf("\nRUNNING RKNN\n");
    int ret = 0;
    TIMER timer, timer_iner;
    timer.start();
    for (int i=0; i<loop_time; i++){
        timer_iner.start();
        ret = run_rknn_app(rknn_app_ctx, input_buffer, output_buffer);
        timer_iner.stop();
        printf("  loop[%d] time: %f ms\n", i, timer_iner.get_time());
    }
    timer.stop();
    printf("Average time: %f ms\n", timer.get_time() / loop_time);
    return ret;
}


int post_process_check_consine_similarity(rknn_app_context_t* rknn_app_ctx,
                                          rknn_app_buffer* output_buffer,
                                          const char* output_folder, 
                                          const char* golden_folder,
                                        Json::Value &root)
{
    int ret = 0;
    printf("\nCHECK OUTPUT\n");
    printf("  check all output to '%s'\n", output_folder);
    printf("  with golden data in '%s'\n", golden_folder);
    // ret = folder_mkdirs(output_folder);

    float* temp_data = NULL;
    rknn_tensor_attr* attr = NULL;
    for (int idx=0; idx< rknn_app_ctx->n_output; idx++){
        attr = &(rknn_app_ctx->out_attr[idx]);

        char * name_strings = (char*)malloc(200);
        memset(name_strings, 0, 200);
        int i = 0;
        while (attr->name[i] != '\0') {
            if (!((attr->name[i] >= '0' && attr->name[i] <= '9') ||
                (attr->name[i] >= 'a' && attr->name[i] <= 'z') ||
                (attr->name[i] >= 'A' && attr->name[i] <= 'Z'))) {
                name_strings[i] = '_';
            }
            else {
                name_strings[i] = attr->name[i];
            }
            i++;
        }

        printf("  output[%d] - %s:\n", idx, name_strings);
        char* golden_path = get_output_path(name_strings, golden_folder);
        char* output_path = get_output_path(name_strings, output_folder);
        temp_data = (float*)rknn_app_unwrap_output_buffer(rknn_app_ctx, &output_buffer[idx], RKNN_TENSOR_FLOAT32, idx);
        
        // save_npy(output_path, temp_data, attr);
        root["name"] = name_strings;
        root["shape"] = Json::arrayValue;
        root["data"];

        for (uint32_t colIdx = 0; colIdx < attr->n_dims; ++colIdx)
        {
            root["shape"][colIdx] = attr->dims[colIdx];
        }

        enum {
            INFERENCE_NONE,
            INFERENCE_DEEPFACE_GENDER,
            INFERENCE_DEEPFACE_AGE,
            INFERENCE_DEEPFACE_RACE,
            INFERENCE_DEEPFACE_EMOTION,
        };
        // TODO: make this a parameter to the function
        int inference_type = INFERENCE_DEEPFACE_GENDER;

        switch (inference_type)
        {
            case INFERENCE_DEEPFACE_GENDER:
                assert(attr->n_dims == 2);
                assert(attr->dims[0] == 1);
                assert(attr->dims[1] == 2);

                root["data"]["gender"]["prob_female"] = temp_data[0];
                root["data"]["gender"]["prob_male"] = temp_data[1];

                break;

            default:
                // General algorithm to write an n-dimensional array to json
                size_t currentIdx = 0;
                Json::Value &walk = root["data"];
                for (size_t colIdx=0; colIdx < attr->n_dims; colIdx++)
                {
                    walk[(int)colIdx] = Json::arrayValue;
                    for (size_t cellIdx=0; cellIdx<attr->dims[colIdx]; cellIdx++)
                    {
                        root["data"][(int)colIdx][(int)cellIdx] = temp_data[currentIdx++];
                    }
                    walk = walk[(int)colIdx];
                }
        }



        free(temp_data);

        if (access(golden_path, F_OK) == 0){
            float cosine_similarity = compare_npy_cos_similarity(golden_path, output_path, attr->n_elems);
            printf("    cosine similarity: %f\n", cosine_similarity);
        }
        else{
            printf("    Ignore compute cos. Golden data '%s' not found\n", golden_path);
        }

        free(name_strings);
        free(golden_path);
        free(output_path);
    }
    return ret;
}


int get_data_shapes(const char* input_path, rknn_tensor_attr* shape_container){
    int ret = 0;
    if (input_path == NULL){
        return -1;
    }
    std::vector<std::string> input_paths_split;
    input_paths_split = split(input_path, "#");
    if (input_paths_split.size() == 0){
        return -1;
    }
    for (int i = 0; i < input_paths_split.size(); i++){
        if (strstr(input_paths_split[i].c_str(), ".npy")) {
            ret = load_npy_shape(input_paths_split[i].c_str(), &shape_container[i]);
            if (ret != 0){
                printf("ERROR: load input '%s' shape failed\n", input_paths_split[i].c_str());
                return -1;
            }
            shape_container[i].fmt = RKNN_TENSOR_NCHW;
        }
        else {
            ret = load_image_shape(input_paths_split[i].c_str(), &shape_container[i]);
            if (ret != 0){
                printf("ERROR: load input '%s' shape failed\n", input_paths_split[i].c_str());
                return -1;
            }
        }
    }
    return 0;
}


int main(int argc, char* argv[]){
    if (argc < 2 ){
        printf("Usage: ./rknn_app_demo model_path image_dir repeat_times\n");
        printf("  Example: ./rknn_app_demo ./Age.rknn ./data/ 20\n");
        return -1;
    }

#ifdef USE_INOTIFY
 	struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = cleanupSigHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGTERM, &sigIntHandler, NULL);
#endif
    int ret = 0;

    // TODO: make model_path optional. if no path given, loop over all models and create one .out.json file
    std::vector<std::string> models {
        "facial_attribute.Gender.rknn",
        // "facial_attribute.Age.rknn",
        // "facial_attribute.Race.rknn",
        // "facial_attribute.Emotion.rknn",
    };

    // init rknn_app
    rknn_app_context_t Age_ctx;
    memset(&Age_ctx, 0, sizeof(rknn_app_context_t));
    const char* model_path = argv[1];
    std::string inpath (argv[2]);
    bool verbose_log = true;
    ret = init_rknn_app(&Age_ctx, model_path, verbose_log);
    if (ret != 0){
        printf("init rknn app failed\n");
        return -1;
    }


#ifdef USE_INOTIFY
    int inotifyFd, wd;
    char buf[BUF_LEN] __attribute__((aligned(8)));
    ssize_t numRead;
    struct inotify_event *event;

    inotifyFd = inotify_init();
    if (inotifyFd == -1)
    {
        perror("inotify_init");
        return NULL;
    }

    wd = inotify_add_watch(inotifyFd, infolder.c_str(), DASH_EVENT_MASK);
    if (wd == -1)
    {
        perror("inotify_add_watch");
        return NULL;
    }

    time_t prev = 0;

    while (!g_quit)
    {
        int return_value;
        fd_set descriptors;
        struct timeval time_to_wait;

        FD_ZERO(&descriptors);
        FD_SET(inotifyFd, &descriptors);

        time_to_wait.tv_sec = 1;
        time_to_wait.tv_usec = 0;

        return_value = select(inotifyFd + 1, &descriptors, NULL, NULL, &time_to_wait);

        if (return_value < 0)
        {
            /* Error */
            sleep(1); // don't know what to do here, coding error, which should never get committed.
            continue;
        }
        else if (!return_value)
        {
            //continue;  // do not continue, need to service the init segs.
        }
        else if (FD_ISSET(inotifyFd, &descriptors))
        {
            /* Process the inotify events */
            numRead = read(inotifyFd, buf, BUF_LEN);
            if (numRead == 0)
                perror("read() from inotify fd returned 0!");
            else if (numRead == -1)
                perror("read");
            else
            {
                for (char *p = buf; p < buf + numRead;)
                {
                    event = (struct inotify_event *)p;
                    if (event->len > 0)
                    {
                        std::string ps(event->name);
                        size_t pos = ps.find_last_of(".");
                        if (pos != std::string::npos)
                        {
                            if (ps.compare(pos + 1, 3, "tmp") != 0) // ffmpeg will write/close tmp files, ignore.
                            {
                                std::string s(infolder);
                                s.append("/");
                                s.append(ps);
//                                fprintf(stderr, "Inot: %s\n", s.c_str());
#else
                                std::string s(inpath);
                                std::string ps(basename(s.c_str()));
#endif
                                Json::Value root;
                                Json::StreamWriterBuilder builder;
                                builder["commentStyle"] = "None";
                                builder["indentation"] = "";
                                builder["precision"] = 2;
#ifdef DYNAMIC_SHAPE_COMPATABLE
                                // set input shape
                                rknn_tensor_attr shape_container[Age_ctx.n_input];
                                memset(shape_container, 0, sizeof(rknn_tensor_attr) * Age_ctx.n_input);
                                ret = get_data_shapes(s.c_str(), shape_container);
                                if (ret < 0){
                                    ret = rknn_app_switch_dyn_shape(&Age_ctx, Age_ctx.in_attr);
                                }
                                else{
                                    ret = rknn_app_switch_dyn_shape(&Age_ctx, shape_container);
                                }
                                if (ret != 0){
                                    printf("set input shape failed\n");
                                    return -1;
                                }
#endif

                                // init input output buffer
                                rknn_app_buffer Age_input_buffer[Age_ctx.n_input];
                                rknn_app_buffer Age_output_buffer[Age_ctx.n_output];
                                ret = init_rknn_app_input_output_buffer(&Age_ctx, Age_input_buffer, Age_output_buffer, false);
                                if (ret != 0){
                                    printf("init input output buffer failed\n");
                                    return -1;
                                }

                                // load input data and wrap to input buffer
                                ret = load_input_data(&Age_ctx, s.c_str(), Age_input_buffer);
                                int input_given = ret;

                                // inference
                                int loop_time = 1;
                                if (argc > 3){
                                    loop_time = atoi(argv[3]);
                                }
                                ret = loop_run(&Age_ctx, Age_input_buffer, Age_output_buffer, loop_time);
                                if (ret != 0){
                                    printf("run rknn app failed\n");
                                    goto out;
                                }

                                // save output result as npy
                                if ((argc > 2) && (input_given>-1)){
                                    ret = post_process_check_consine_similarity(&Age_ctx, Age_output_buffer, inpath.c_str(), "./data/outputs/golden", root);
                                }
                                else{
                                    printf("Inputs was not given, skip save output\n");
                                }

                                {
                                    std::string response(Json::writeString(builder, root));

                                    std::string resultDataPath(inpath + ".out.json");
                                    std::ofstream dataFileOut(resultDataPath.c_str());

                                    if (dataFileOut.is_open())
                                    {
                                        dataFileOut << response;
                                    }
                                    else
                                    {
                                        printf("Error writing data file\n");
                                    }
                                }
#ifdef USE_INOTIFY
                            }
                        }
                    }
                    p += sizeof(struct inotify_event) + event->len;
                }
            }
        }
    }
#endif
out:
#ifdef USE_INOTIFY
    inotify_rm_watch(inotifyFd, wd);
#endif

    // release rknn
    release_rknn_app(&Age_ctx);

    return 0;
}