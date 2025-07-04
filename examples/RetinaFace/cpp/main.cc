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

#include "retinaface.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_retinaface_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0) {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        return -1;
    }

    retinaface_result result;
    ret = inference_retinaface_model(&rknn_app_ctx, &src_image, &result);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d\n", ret);
        goto out;
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

        write_image(std::string(std::string("face-") + std::to_string(i) + ".jpg").c_str(), &face_image);

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
    }
    write_image("result.jpg", &src_image);

out:
    ret = release_retinaface_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_retinaface_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL) {
        free(src_image.virt_addr);
    }

    return 0;
}
