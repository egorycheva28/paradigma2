// paradigma4.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <time.h>
#include <intrin.h>

using namespace std;
using namespace cv;

void mosaic(Mat& image, int mosaicSize)
{
#pragma omp parallel for
    for (int i = 0; i < image.rows; i += mosaicSize)
    {
        for (int j = 0; j < image.cols; j += mosaicSize)
        {
            Vec3i averageColor = Vec3i(0, 0, 0);

            for (int x = i; x < min(i + mosaicSize, image.rows); x++)
            {
                for (int y = j; y < min(j + mosaicSize, image.cols); y++)
                {
                    Vec3b pixel = image.at<Vec3b>(x, y);
                    averageColor += Vec3i(pixel[0], pixel[1], pixel[2]);
                }
            }

            averageColor /= mosaicSize * mosaicSize;

            for (int x = i; x < min(i + mosaicSize, image.rows); x++)
            {
                for (int y = j; y < min(j + mosaicSize, image.cols); y++)
                {
                    image.at<Vec3b>(x, y) = Vec3b(averageColor[0], averageColor[1], averageColor[2]);
                }
            }
        }
    }
}

void contrast(Mat& image)
{
#pragma omp parallel for 
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            for (int c = 0; c < image.channels(); c++)
            {
                int pixel = (image.at<Vec3b>(i, j)[c] - 128) * 0.5 + 128;
                image.at<Vec3b>(i, j)[c] = max(0, min(255, pixel));
            }
        }
    }
}

void vectorizationMosaica(Mat& image, int mosaicSize)
{
    for (int i = 0; i < image.rows; i += mosaicSize)
    {
        for (int j = 0; j < image.cols; j += mosaicSize)
        {
            __m128i sum = _mm_setzero_si128();

            for (int x = 0; x < mosaicSize; x++)
            {
                for (int y = 0; y < mosaicSize; y++)
                {
                    if (i + x < image.rows && j + y < image.cols)
                    {
                        Vec3b pixel = image.at<Vec3b>(i + x, j + y);
                        __m128i color = _mm_setr_epi8(pixel[0], pixel[1], pixel[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                        sum = _mm_add_epi32(sum, _mm_unpacklo_epi16(_mm_unpacklo_epi8(color, _mm_setzero_si128()), _mm_setzero_si128()));
                    }
                }
            }

            sum = _mm_srli_epi32(sum, 6);
            Vec3b averageColor = Vec3b(_mm_extract_epi16(sum, 0), _mm_extract_epi16(sum, 2), _mm_extract_epi16(sum, 4));
            
            for (int x = 0; x < mosaicSize; x++)
            {
                for (int y = 0; y < mosaicSize; y++)
                {
                    if (i + x < image.rows && j + y < image.cols)
                    {
                        image.at<Vec3b>(i + x, j + y) = averageColor;
                    }
                }
            }
        }
    }
}

void vectorizationContrast(Mat image, Mat& newImage, float contrast) 
{
    float coef = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast));

    union
    {
        __m256 img;
        __m256 pixel;
        float res[8];
        float result[8];
    };

    for (int x = 0; x < image.rows; x++)
    {
        for (int y = 0; y < image.cols - 8; y += 8)
        {
            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < 8; i++)
                {
                    res[i] = image.at<Vec3b>(y + i, x)[c];
                }

                pixel = _mm256_add_ps(_mm256_set1_ps(128), _mm256_mul_ps(_mm256_set1_ps(coef), _mm256_sub_ps(img, _mm256_set1_ps(128))));
                pixel = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(_mm256_set1_ps(255), pixel));

                for (int i = 0; i < 8; i++)
                {
                    newImage.at<Vec3b>(y + i, x)[c] = (int)result[i];
                }
            }
        }
    }
}

int main()
{
    clock_t start = clock();

    Mat image = imread("300x300.png");

    //vectorizationContrast(image, image, -100.0);
    //contrast(image);
    //imwrite("contrast300.png", image);

    //vectorizationMosaica(image, 7);
    //mosaic(image, 7);
    //imwrite("mosaica2400.png", image);

    clock_t end = clock();
    double time = (double)(end - start) / CLOCKS_PER_SEC;
    cout << time;
}

/*
#include <intrin.h>
векторизация
void vectorisazia()

*/

/*vector<Vec3b>copy1 = copy(image);
struct RGB
{
    int R, G, B;
};

vector<Vec3b>copy(Mat& image)
{
    vector<Vec3b>pixels(image.rows * image.cols);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            pixels.push_back(image.at<Vec3b>(i, j));
        }
    }
    return pixels;
}
void contrast1(vector<Vec3b>& copy1)
{
    for (int i = 0; i < copy1.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int g = (copy1[i][j] - 128) * 0.5 + 128;
            copy1[i][j] = max(0, min(255, g));
        }
    }
}*/

/*#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void applyMosaic(const char* inputFileName, const char* outputFileName) {
  int width, height, channels;
  unsigned char* image = stbi_load(inputFileName, &width, &height, &channels, 3);

  if (image == NULL) {
    std::cerr << "Error loading image" << std::endl;
    return;
  }

  int mosaicSize = 7;

  for (int y = 0; y < height; y += mosaicSize) {
    for (int x = 0; x < width; x += mosaicSize) {
      int sumR = 0, sumG = 0, sumB = 0;
      int count = 0;

      for (int j = y; j < std::min(y + mosaicSize, height); j++) {
        for (int i = x; i < std::min(x + mosaicSize, width); i++) {
          int index = (j * width + i) * 3;
          sumR += image[index];
          sumG += image[index + 1];
          sumB += image[index + 2];
          count++;
        }
      }

      int avgR = sumR / count;
      int avgG = sumG / count;
      int avgB = sumB / count;

      for (int j = y; j < std::min(y + mosaicSize, height); j++) {
        for (int i = x; i < std::min(x + mosaicSize, width); i++) {
          int index = (j * width + i) * 3;
          image[index] = avgR;
          image[index + 1] = avgG;
          image[index + 2] = avgB;
        }
      }
    }
  }

  stbi_write_png(outputFileName, width, height, 3, image, width * 3);

  stbi_image_free(image);
}
int main() {
  applyMosaic("input.png", "output.png");
  return 0;
}*/
/*int main()
{
  Mat img(400, 500, CV_8UC3);
  string text = "Hello world!";
  Point textOrg(100, img.rows / 2);
  int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 2;
  Scalar color(200, 100, 50);
  putText(img, text, textOrg, fontFace, fontScale, color);
  imshow("My World", img);
  waitKey(0);
  return 0;
}*/

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
