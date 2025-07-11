// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/inotify.h>
#include <signal.h>
#include <limits.h>
#include <sys/stat.h>
#include <json/json.h>
#include <fstream>

#include "retinaface.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"

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
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_path> <input_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    std::string inpath (argv[2]);

#ifdef USE_INOTIFY
    std::string outfolder ("data/");
    std::string outfacefolder ( outfolder + "faces/");

    mkdir(outfolder.c_str(), 666);
    mkdir(outfacefolder.c_str(), 666);

	struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = cleanupSigHandler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGTERM, &sigIntHandler, NULL);
#endif

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_retinaface_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, model_path);
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

    wd = inotify_add_watch(inotifyFd, inpath.c_str(), DASH_EVENT_MASK);
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
                                std::string s(inpath);
                                s.append("/");
                                s.append(ps);
//                                fprintf(stderr, "Inot: %s\n", s.c_str());
#else
                                std::string s(inpath);
                                std::string ps(basename(s.c_str()));
                                std::string outfacefolder;
#endif

                                Json::Value root;
                                Json::StreamWriterBuilder builder;
                                builder["commentStyle"] = "None";
                                builder["indentation"] = "";
                                builder["precision"] = 2;

                                std::string resultFile(outfacefolder + ps + ".out");
                                std::string resultImgPath(resultFile + ".jpg");
                                std::string resultDataPath(resultFile + ".json");

                                image_buffer_t src_image;
                                memset(&src_image, 0, sizeof(image_buffer_t));
                                ret = read_image(s.c_str(), &src_image);
                                if (ret != 0) {
                                    printf("read image fail! ret=%d image_path=%s\n", ret, s.c_str());
                                    return -1;
                                }

                                retinaface_result result;
                                ret = inference_retinaface_model(&rknn_app_ctx, &src_image, &result);
                                if (ret != 0) {
                                    printf("init_retinaface_model fail! ret=%d\n", ret);
                                    goto out;
                                }
                                if (result.count > 0)
                                {
                                    root["faces"] = Json::arrayValue;
                                }

                                for (int i = 0; i < result.count; ++i) {
                                    int rx = result.object[i].box.left;
                                    int ry = result.object[i].box.top;
                                    int rw = result.object[i].box.right - result.object[i].box.left;
                                    int rh = result.object[i].box.bottom - result.object[i].box.top;

                                    // save a copy of the face
                                    image_buffer_t face_image;
                                    memset(&face_image, 0, sizeof(image_buffer_t));
                                    face_image.format = IMAGE_FORMAT_RGB888;
                                    face_image.size = 224*224*3;
                                    face_image.virt_addr = (unsigned char *)malloc(face_image.size);
                                    face_image.height = 224;
                                    face_image.width = 224;

                                    image_rect_t src_box;
                                    src_box.left = rx;
                                    src_box.top = ry;
                                    src_box.right = rx+rw;
                                    src_box.bottom = ry+rh;

                                    printf("Src box: %d %d %d %d width: %d height: %d\n", src_box.left, src_box.top, src_box.right, src_box.bottom, rw, rh);

                                    // The detected face region may not be a square, but the model requires a square input, so make sure the aspect ratio is preserved.
                                    image_rect_t dst_box;
                                    dst_box.left =   (rw < rh ? (224-(224*rw/rh))/2 : 0);
                                    dst_box.right =  (rw < rh ? dst_box.left + (224*rw/rh) : 224);
                                    dst_box.top =    (rw > rh ? (224-(224*rh/rw))/2 : 0);
                                    dst_box.bottom = (rw > rh ? dst_box.top + (224*rh/rw) : 224);

                                    printf("Dst box: %d %d %d %d\n", dst_box.left, dst_box.top, dst_box.right, dst_box.bottom);

                                    convert_image(&src_image, &face_image, &src_box, &dst_box, 0);
                                    std::string faceImgPath(resultFile + "." + "face-" + std::to_string(i) + ".jpg");
                                    write_image(faceImgPath.c_str(), &face_image);

                                    free(face_image.virt_addr);

                                    // draw on original
                                    draw_rectangle(&src_image, rx, ry, rw, rh, COLOR_GREEN, 3);
                                    char score_text[20];
                                    snprintf(score_text, 20, "%0.2f", result.object[i].score);
                                    printf("face @(%d %d %d %d) score=%f\n", result.object[i].box.left, result.object[i].box.top,
                                        result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
                                    draw_text(&src_image, score_text, rx+rw, ry+rh, COLOR_RED, 12);
                                    for(int j = 0; j < 5; j++) {
                                        draw_circle(&src_image, result.object[i].ponit[j].x, result.object[i].ponit[j].y, 2, COLOR_ORANGE, 4);
                                    }

                                    root["faces"][i]["idx"] = i;
                                    root["faces"][i]["width"] = rw;
                                    root["faces"][i]["height"] = rh;
                                    root["faces"][i]["left"] = rx;
                                    root["faces"][i]["top"] = ry;
                                    root["faces"][i]["right"] = result.object[i].box.right;
                                    root["faces"][i]["bottom"] = result.object[i].box.bottom;
                                    root["faces"][i]["imgPath"] = faceImgPath;
                                    root["faces"][i]["confidence"] = score_text;
                                }
                                write_image(resultImgPath.c_str(), &src_image);
                                root["imgPath"] = resultImgPath;

                                {

                                    std::string response(Json::writeString(builder, root));

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


                                if (src_image.virt_addr != NULL) {
                                    free(src_image.virt_addr);
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

    ret = release_retinaface_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_retinaface_model fail! ret=%d\n", ret);
    }

    return 0;
}
