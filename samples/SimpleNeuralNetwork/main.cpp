//
// Created by Steven Roddan on 7/16/2025.
//

#include <iomanip>

#include "SixClassDataGenerator.h"
#include "ThreeClassDataGenerator.h"
#include "SimpleNeuralNetwork.h"

#include <Windows.h>
#include <vector>
#include <cmath>

#define ID_REFRESH_BUTTON 1

static int WIDTH = 1600;
static int HEIGHT = 1200;
static std::vector<unsigned int> g_pixels;   // 0x00RRGGBB (BGRA in memory)
static BITMAPINFO g_bi = {};

#define MAX_EPOCH 10

std::vector<DataPoint> dp_input, dp_truth_validation, six_class_input, six_class_truth_validation;
NeuralNetwork three_class_nn({2, 12, 24, 3}, .075f);
NeuralNetwork six_class_nn({2, 64, 128, 6}, .05f);

inline void putPixel(int x, int y, unsigned int rgb)
{
    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return;
    g_pixels[y * WIDTH + x] = rgb; // 0x00RRGGBB
}

static void clear(unsigned int rgb)
{
    std::fill(g_pixels.begin(), g_pixels.end(), rgb);
}

static void drawAxes()
{
    // light gray
    const unsigned int axis = 0x00A0A0A0;
    int cx = WIDTH / 2;
    int cy = HEIGHT / 2;

    for (int x = 0; x < WIDTH; ++x) putPixel(x, cy, axis);
    for (int y = 0; y < HEIGHT; ++y) putPixel(cx, y, axis);
}

static void drawThreeClassInput(const DataPoint& dp, int guess, int xOffset, int yOffset) {
    const double scaleY = HEIGHT / 4;
    const double periodPx = WIDTH / 4;
    int color = 0x00000000;
    if(guess != dp.label) {
        color = 0x00dddddd;
    }
    else if(dp.label == 0) {
        color = 0x00ff0000;
    } else if(dp.label == 1) {
        color = 0x0000ff00;
    } else {
        color = 0x000000ff;
    }
    putPixel(dp.x * periodPx + (WIDTH / 4) + xOffset,(-dp.y) * scaleY + (HEIGHT / 4) + yOffset, color);
}

static void drawSixClassInput(const DataPoint& dp, int guess, int xOffset, int yOffset) {
    const double scaleY = HEIGHT / 4;
    const double periodPx = WIDTH / 4;

    int color = 0x00000000;

    if (guess != dp.label) {
        // Wrong prediction → light gray
        color = 0x00dddddd;
    } else {
        // 6 classes, assign distinct colors
        switch (dp.label) {
            case 0: color = 0x00ff0000; break; // red
            case 1: color = 0x0000ff00; break; // green
            case 2: color = 0x000000ff; break; // blue
            case 3: color = 0x00ffff00; break; // yellow
            case 4: color = 0x00ff00ff; break; // magenta
            case 5: color = 0x0000ffff; break; // cyan
            default: color = 0x00000000; break;
        }
    }

    // Transform data point to pixel coords
    int px = static_cast<int>(dp.x * periodPx + (WIDTH / 4) + xOffset);
    int py = static_cast<int>((-dp.y) * scaleY + (HEIGHT / 4) + yOffset);

    putPixel(px, py, color);
}

static void drawThreeClassInput(const std::vector<DataPoint>& input) {
    const double scaleY = HEIGHT / 4;
    const double periodPx = WIDTH / 4;
    for(const DataPoint& dp : input) {
        int color = 0x00000000;
        if(dp.label == 0) {
            color = 0x00ff0000;
        } else if(dp.label == 1) {
            color = 0x0000ff00;
        } else {
            color = 0x000000ff;
        }
        putPixel(dp.x * periodPx + (WIDTH / 4),(-dp.y) * scaleY + (HEIGHT / 4),color);
    }
}

static void drawSixClassInput(const std::vector<DataPoint>& input, uint64_t xOffset, uint64_t yOffset) {
    const double scaleY = HEIGHT / 4;
    const double periodPx = WIDTH / 4;
    for(const DataPoint& dp : input) {
        int color;
        switch (dp.label) {
            case 0: color = 0x00ff0000; break; // red
            case 1: color = 0x0000ff00; break; // green
            case 2: color = 0x000000ff; break; // blue
            case 3: color = 0x00ffff00; break; // yellow
            case 4: color = 0x00ff00ff; break; // magenta
            case 5: color = 0x0000ffff; break; // cyan
            default: color = 0x00000000; break;
        }
        putPixel(dp.x * periodPx + (WIDTH / 4) + xOffset,(-dp.y) * scaleY + (HEIGHT / 4) + yOffset, color);
    }
}

float mapRange(float oldValue, float oldMin, float oldMax, float newMin, float newMax) {
    float oldRange = oldMax - oldMin;
    if (oldRange == 0.0f) return newMin; // avoid division by zero
    float newRange = newMax - newMin;
    float newValue = ((oldValue - oldMin) * newRange) / oldRange + newMin;
    return newValue;
}

static void drawThreeClassProblem(const std::vector<DataPoint>& threeClassinput, const std::vector<DataPoint>& threeClassValidation,
                                  NeuralNetwork& three_nn) {
    NaNLA::Common::ThreadPool tp(2);
    std::vector<std::future<void>> futures;
    futures.emplace_back(tp.queue([&threeClassinput] {drawThreeClassInput(threeClassinput);}));

    const int range = 2;
    const int finalWidth = WIDTH / 2;
    const int finalHeight = HEIGHT / 2;

    futures.emplace_back(tp.queue([&] {
        std::vector<std::vector<float>> batchData;
        std::vector<std::vector<float>> batchLabels;

        for (const DataPoint &dp: threeClassinput) {

            batchData.push_back({dp.x, dp.y});
            batchLabels.push_back({
                                          dp.label == 0 ? 1.0f : 0.0f,
                                          dp.label == 1 ? 1.0f : 0.0f,
                                          dp.label == 2 ? 1.0f : 0.0f
                                  });
        }
        for(int e = 0; e < MAX_EPOCH; e++) {
            three_nn.train(batchData, batchLabels);
        }

        // Build batch input
        std::vector<std::vector<float>> batchInputs;
        for (const DataPoint &dp : threeClassValidation) {
            batchInputs.push_back({dp.x, dp.y});
        }

        // Run batch prediction
        auto batchPreds = three_nn.predict(batchInputs);

        int totalRight = 0;

        // Loop over the batch results
        for (size_t i = 0; i < threeClassValidation.size(); i++) {
            const DataPoint &dp = threeClassValidation[i];
            const auto &pred = batchPreds[i];

            // Find predicted class
            auto pLabel = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));

            if (pLabel == dp.label)
                totalRight++;

            // Draw each sample
            drawThreeClassInput(dp, pLabel, WIDTH / 2, 0);
        }
        std::cout << "Three Class Validation %: " << std::fixed << std::setprecision(4) << (((float) totalRight) / ((float) threeClassValidation.size())) * 100.0
                  << std::endl;
    }));

    for(std::future<void>& f : futures) {
        f.get();
    }
}

static void drawSixClassProblem(const std::vector<DataPoint>& sixClassinput, const std::vector<DataPoint>& sixClassValidation,
                                  NeuralNetwork& six_nn) {
    NaNLA::Common::ThreadPool tp(2);
    std::vector<std::future<void>> futures;
    futures.emplace_back(tp.queue([&sixClassinput] {drawSixClassInput(sixClassinput, 0, HEIGHT / 2);}));

    futures.emplace_back(tp.queue([&] {
        std::vector<std::vector<float>> batchData;
        std::vector<std::vector<float>> batchLabels;

        for (const DataPoint &dp: sixClassinput) {

            batchData.push_back({dp.x, dp.y});
            batchLabels.push_back({
                                          dp.label == 0 ? 1.0f : 0.0f,
                                          dp.label == 1 ? 1.0f : 0.0f,
                                          dp.label == 2 ? 1.0f : 0.0f,
                                          dp.label == 3 ? 1.0f : 0.0f,
                                          dp.label == 4 ? 1.0f : 0.0f,
                                          dp.label == 5 ? 1.0f : 0.0f
                                  });
        }
        for(int e = 0; e < MAX_EPOCH; e++) {
            six_nn.train(batchData, batchLabels);
        }


        // Build batch input
        std::vector<std::vector<float>> batchInputs;
        for (const DataPoint &dp : sixClassValidation) {
            batchInputs.push_back({dp.x, dp.y});
        }

        // Run batch prediction
        auto batchPreds = six_nn.predict(batchInputs);

        uint64_t totalRight = 0;

        // Loop over the batch results
        for (size_t i = 0; i < sixClassValidation.size(); i++) {
            const DataPoint &dp = sixClassValidation[i];
            const auto &pred = batchPreds[i];

            // Find predicted class
            auto pLabel = std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));

            if (pLabel == dp.label)
                totalRight++;

            // Draw each sample
            drawSixClassInput(dp, pLabel, WIDTH / 2, HEIGHT / 2);
        }

        // Print accuracy
        std::cout << "Six Class Validation %: "
                  << std::fixed << std::setprecision(4)
                  << (static_cast<float>(totalRight) / static_cast<float>(sixClassValidation.size())) * 100.0
                  << std::endl;
    }));

    for(std::future<void>& f : futures) {
        f.get();
    }
}

static void draw(const std::vector<DataPoint>& threeClassinput, const std::vector<DataPoint>& threeClassValidation,
                 const std::vector<DataPoint>& sixClassinput, const std::vector<DataPoint>& sixClassValidation,
                 NeuralNetwork& three_nn, NeuralNetwork& six_nn) {

    NaNLA::Common::ThreadPool tp(2);
    std::vector<std::future<void>> futures;
    futures.emplace_back(tp.queue([&]{drawThreeClassProblem(threeClassinput, threeClassValidation, three_nn);}));
    futures.emplace_back(tp.queue([&]{drawSixClassProblem(sixClassinput, sixClassValidation, six_nn);}));

    for(auto& f : futures) {
        f.get();
    }
    drawAxes();
}

static void renderScene(const std::vector<DataPoint>& three_class_input, const std::vector<DataPoint>& three_class_truth_validation,
                        const std::vector<DataPoint>& six_class_input, const std::vector<DataPoint>& six_class_truth_validation,
                        NeuralNetwork& three_class_nn, NeuralNetwork& six_class_nn, bool drawTrain = false)
{
    draw(three_class_input, three_class_truth_validation, six_class_input, six_class_truth_validation, three_class_nn, six_class_nn);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
        case WM_CREATE:
            g_pixels.resize(WIDTH * HEIGHT);

            // Prepare BITMAPINFO for top-down 32-bit DIB (no alpha)
            ZeroMemory(&g_bi, sizeof(g_bi));
            g_bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            g_bi.bmiHeader.biWidth = WIDTH;
            g_bi.bmiHeader.biHeight = -HEIGHT;         // negative = top-down
            g_bi.bmiHeader.biPlanes = 1;
            g_bi.bmiHeader.biBitCount = 32;            // BGRA in memory
            g_bi.bmiHeader.biCompression = BI_RGB;

            clear(0x00222222);
            renderScene(dp_input, dp_truth_validation, six_class_input, six_class_truth_validation, three_class_nn, six_class_nn);

            SetTimer(hWnd, 1, 25, NULL);

            return 0;
        case WM_COMMAND:
            if (LOWORD(wParam) == 1) { // button ID
                // User clicked button
                // Trigger repaint
                InvalidateRect(hWnd, NULL, TRUE);
            }
            break;
        case WM_TIMER:
            // Trigger repaint every tick
            InvalidateRect(hWnd, NULL, FALSE);
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);

            // Blit the pixel buffer to the window
            StretchDIBits(
                    hdc,
                    0, 0, WIDTH, HEIGHT,          // dest rect
                    0, 0, WIDTH, HEIGHT,          // src rect
                    g_pixels.data(),
                    &g_bi,
                    DIB_RGB_COLORS,
                    SRCCOPY
            );

            renderScene(dp_input, dp_truth_validation, six_class_input, six_class_truth_validation, three_class_nn, six_class_nn, true);

            EndPaint(hWnd, &ps);
            return 0;
        }

        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) PostQuitMessage(0);
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    dp_input = generate_dataset(12800, 1.0f);
    dp_truth_validation = generate_dataset(6400, 1.0f);

    six_class_input = generate_checkerboard(12800, 1.0f);
    six_class_truth_validation = generate_checkerboard(6400, 1.0f);

    const wchar_t CLASS_NAME[] = L"PixelGraphWindow";

    WNDCLASS wc = {};
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInstance;
    wc.lpszClassName = reinterpret_cast<LPCSTR>(CLASS_NAME);
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);

    RegisterClass(&wc);

    DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_MAXIMIZEBOX | WS_THICKFRAME);
    RECT r = { 0,0, WIDTH, HEIGHT };
    AdjustWindowRect(&r, style, FALSE);

    HWND hWnd = CreateWindowEx(
            0, reinterpret_cast<LPCSTR>(CLASS_NAME), reinterpret_cast<LPCSTR>("Pixel Graph (ESC to quit)"), style,
            CW_USEDEFAULT, CW_USEDEFAULT,
            r.right - r.left, r.bottom - r.top,
            nullptr, nullptr, hInstance, nullptr
    );

    // Create the Refresh button
    HWND hButton = CreateWindowEx(
            0, reinterpret_cast<LPCSTR>("BUTTON"), reinterpret_cast<LPCSTR>("Next Epoch"),
            WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            10, 10, 100, 30,       // position & size
            hWnd, (HMENU)ID_REFRESH_BUTTON, hInstance, nullptr
    );

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return (int)msg.wParam;
}