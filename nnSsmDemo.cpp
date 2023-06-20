#define GL_GLEXT_PROTOTYPES
#define EGL_EGLEXT_PROTOTYPES
#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <string.h>
#include <sstream>
#include <fstream>
#include <optional.hpp>
#include <ssm.h>
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C"
{
//#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
}

#include <Graph2DRenderer.h>
#include <vectorgraph.h>
#include <wulipvector.h>
#include <Size.h>
#include <foreach.h>
#include <Animation.h>
#include <fatum.h>
#include <dbgraph.h>
#include <Shape.h>
#include <randomgraph.h>
#include <kmeans.h>
#include <incrementaldelaunay.h>
#include <quadtree.h>
#include <louvain.h>

#include <dikjstra.h>
#include <paths.h>

#include <text.h>

#include <highfive/H5Easy.hpp>
/*

related :
https://core.ac.uk/download/pdf/82619396.pdf

*/

using namespace std;
using namespace wlp;

//std::experimental::optional<wlp::Mark> selectedMark;
wlp::Fatum *dbVis = 0;
easingFunction easeFunc = easeLinear;
static int WIDTH = 512;
static int HEIGHT = 512;

static bool drawLabels = true;
// VectorGraph graph;
NodeProperty<Vec2d> nodePos;
NodeProperty<Vec2d> oriPos;
EdgeProperty<double> weight;

// NodeProperty<Mark> marks;
//=================================

/**
 * @brief glut_reshape_callback
 * @param width
 * @param height
 */
static void glut_reshape_callback(int width, int height)
{
    dbVis->getCamera().setViewport(Vec4i(0, 0, width, height));
    WIDTH = width;
    HEIGHT = height;
    glutPostRedisplay();
}
/**
 * @brief glut_draw_callback
 */
static void glut_draw_callback(void)
{
    dbVis->renderFrame();
    glutSwapBuffers();
    if (dbVis->needsRendering())
        glutPostRedisplay();
}

/**
 * @brief lastX
 */
static int lastX = -1, lastY = -1;
static int mouseState;

#ifdef EMSCRIPTEN
#include <emscripten/bind.h>
wlp::Fatum *facade()
{
    return dbVis;
}

EMSCRIPTEN_BINDINGS(facade)
{
    emscripten::function("fatum", &facade, emscripten::allow_raw_pointers());
}

#endif

void PPMWriter(unsigned char *in, char *name, int dimx, int dimy) {
    int i, j;
    FILE *fp = fopen(name, "wb"); /* b - binary mode */
    (void) fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
    for (j = dimy - 1; j >= 0; --j) {
        for (i = 0; i < dimx; ++i) {
            static unsigned char color[3];
            color[0] = in[3*i+3*j*dimy];  /* red */
            color[1] = in[3*i+3*j*dimy+1];  /* green */
            color[2] = in[3*i+3*j*dimy+2];  /* blue */
            (void) fwrite(color, 1, 3, fp);
        }
    }
    (void) fclose(fp);
}

// taken from iagenerator
void saveImage() {
    unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char) * 3 * WIDTH * HEIGHT);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, image);
    // Warning : enregistre de bas en haut
    char buffer [100];
    sprintf(buffer, "capture.ppm");
    PPMWriter(image, buffer,WIDTH, HEIGHT);
}

/**
 * @brief keyboard
 * @param key
 * @param x
 * @param y
 */
static void keyboard(const unsigned char key, const int, const int)
{
    switch (int(key))
    {
    case 'c':
        dbVis->center();
        glutPostRedisplay();
        break;
    case 's':
        dbVis->animate();
        glutPostRedisplay();
        break;
    case 'q':
        saveImage();
        cout << "EXIT ... " << endl;
        exit(EXIT_SUCCESS);
    default:
        return;
    }
    glutPostRedisplay();
}
/**
 * @brief glut_special
 * @param special
 * @param crap
 * @param morecrap
 */
static void glut_special(int special, int, int)
{
    switch (special)
    {
    case GLUT_KEY_LEFT:
        break;
    case GLUT_KEY_RIGHT:
        break;
    case GLUT_KEY_UP:
        break;
    case GLUT_KEY_DOWN:
        break;
    case GLUT_KEY_F11:
        break;
    }
    glutPostRedisplay();
}

struct RectM
{
    RectM(wlp::Fatum *dbvis, Rectd rec) : dbvis(dbvis)
    {
        dbvis->defaultMark().color(Color::Bronze).size(Size(0.1, 0.1, 0.1)).shape(wlp::Shape::NONE);
        a = dbvis->addMark().position(rec[0].x(), rec[0].y(), 0);
        b = dbvis->addMark().position(rec[1].x(), rec[0].y(), 0);
        c = dbvis->addMark().position(rec[1].x(), rec[1].y(), 0);
        d = dbvis->addMark().position(rec[0].x(), rec[1].y(), 0);
        ab = dbvis->addConnection(a, b);
        bc = dbvis->addConnection(b, c);
        cd = dbvis->addConnection(c, d);
        da = dbvis->addConnection(d, a);
    }

    void del()
    {
        a.del();
        b.del();
        c.del();
        d.del();
    }

    wlp::Mark a, b, c, d;
    wlp::Connection ab, bc, cd, da;
    wlp::Fatum *dbvis;
};


struct GeneratorParameters {
    int dataSize;
    int spaceDims;
    int superPointCount;
    vector<int> clusterCount;
    int maxClusterCount;
    vector<double> separationFactors;
    vector<double> superPointDataShare;
    vector<double> clusterSizeVariation;
    map<int, double> clusterDensities;
    vector<map<int, map<int, bool>>> clusterExclusions;
    double missclassifiedRate;
};

struct PostGenerationOutlierGenerator {
    int missCount;
    double variationMin;
    double variationMax;
    char* inputFile;
    char* outputFile;
};

void initVis()
{
    dbVis = new wlp::Fatum();
    dbVis->initGL();
    dbVis->getCamera().setViewport(Vec4i(0, 0, WIDTH, HEIGHT));
    dbVis->layerOn(MARKS | TEXT | CONNECTIONS);
}

/**
 * @brief motionFunc
 * @param x
 * @param y
 */
static void motionFunc(int x, int y)
{
    //if (dbVis->needsRendering()) return; //do not treat events until the graph is re rendered (else too many events)
    if (x < 0 || x > WIDTH || y < 0 || y > HEIGHT)
        return;
    if (lastX < 0 || lastX > WIDTH || lastY < 0 || lastY > HEIGHT)
        return;
    if (mouseState == GLUT_LEFT_BUTTON)
    {
        Vec2f current = Vec2f(x, y);
        Vec3f currentModel = dbVis->getCamera().windowToModel(current);
        Vec3f previousModel = dbVis->getCamera().windowToModel(Vec2f(lastX, lastY));
        Vec3f move = currentModel - previousModel;
        dbVis->getCamera().moveModel(move[0], move[1]);
        dbVis->getCamera().swap();
        glutPostRedisplay();
        lastX = x;
        lastY = y;
    }
    else if (mouseState == GLUT_RIGHT_BUTTON)
    {
        Vec2f current = Vec2f(x, y);
        Vec3f currentModel = dbVis->getCamera().windowToModel(current);
        dbVis->swap();
        glutPostRedisplay();
    }
}

static void mouseFunc(int button, int state, int x, int y)
{
    if (x < 0 || x > WIDTH || y < 0 || y > HEIGHT)
        return;
    mouseState = button;
    if (button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN)
        {
            lastX = x;
            lastY = y;
        }
        // wheel up
    }
    else if (button == 3)
    {
        dbVis->getCamera().zoom(1.05, Vec2f(x, y));
        dbVis->getCamera().swap();
        if (dbVis->needsRendering())
            glutPostRedisplay();
    }
    else if (button == 4)
    {
        dbVis->getCamera().zoom(0.95, Vec2f(x, y));
        dbVis->getCamera().swap();
        if (dbVis->needsRendering())
            glutPostRedisplay();
    }
    else if (button == GLUT_RIGHT_BUTTON)
    {
        lastX = x;
        lastY = y;
        if (state == GLUT_DOWN)
        {
            motionFunc(x, y);
        }
    }
}

const unsigned int MAX_SIZE = 1000;
namespace
{
    void tokenize(const string &str, vector<string> &tokens, const string &delimiters)
    {
        if (str.size() < 1)
            return;
        tokens.clear();
        // Skip delimiters at beginning.
        string::size_type lastPos = str.find_first_not_of(delimiters, 0);
        // Find first "non-delimiter".
        string::size_type pos = str.find_first_of(delimiters, lastPos);
        while (string::npos != pos || string::npos != lastPos)
        {
            // Found a token, add it to the vector.
            tokens.push_back(str.substr(lastPos, pos - lastPos));
            // Skip delimiters.  Note the "not_of"
            lastPos = str.find_first_not_of(delimiters, pos);
            // Find next "non-delimiter"
            pos = str.find_first_of(delimiters, lastPos);
        }
    }
}


/**
 * @brief glut_idle_callback
 */
static void glut_idle_callback(void)
{
    glutPostRedisplay();
    if (dbVis->needsRendering())
    {
        glutPostRedisplay();

        saveImage();
        // cout << "EXIT ... " << endl;
        // exit(EXIT_SUCCESS);

    }
}


int loadRNVectorsH5(string vectorFile, map<int, VectorND> &vectors) {
    // char line[MAX_SIZE];
    int index = -1;
    H5Easy::File file(vectorFile, H5Easy::File::ReadWrite);
    int size = H5Easy::load<int>(file, "size");
    while (index < size - 1) {
        ++index;
        vector<double> v = H5Easy::load<vector<double>>(file, to_string(index));
        vectors[index] = VectorND(v.size());
        for (int i = 0; i < v.size(); ++i) {
            vectors[index][i] = v[i];
            vectors[index].id = index;
        }
    }

    return size;

}

void convertRNVectorsToMatrix(string inputFilename, string outputFilename) {
    map<int, VectorND> vectors;
    int size = loadRNVectorsH5(inputFilename, vectors);
    H5Easy::File file(outputFilename, H5Easy::File::Overwrite);
    vector<vector<double>> convertedVectors;
    for (int i = 0; i < size; ++i) {
        vector<double> v;
        for (int j = 0; j < vectors[i].size(); ++j) {
            v.push_back(vectors[i][j]);
        }
        convertedVectors.push_back(v);
    }
    cerr << "done reading" << endl;
    H5Easy::dump(file, "datakey", convertedVectors);
}

void writeGridMapFile(string outputFile, vector<map<int, map<int, int>>> &grids, vector<vector<double>> &meanDistances, int gridWidth) {
    H5Easy::File file(outputFile, H5Easy::File::Overwrite);
    string v = "grid";
    H5Easy::dump(file, "projType", v);
    H5Easy::dump(file, "size", gridWidth * gridWidth);
    int layerCount = grids.size();
    H5Easy::dump(file, "layerCount", layerCount);
    for (int i = 0; i < layerCount; ++i) {
        for (int x = 0; x < gridWidth; ++x) {
            vector<int> ycolumn(gridWidth);
            for (int y = 0; y < gridWidth; ++y) {
                ycolumn[y] = grids[i][x][y];
            }

            H5Easy::dump(file, "layers/" + to_string(i) + "/x/" + to_string(x), ycolumn);
        }

        H5Easy::dump(file, "layers/" + to_string(i) + "/meanDistances", meanDistances[i]);
    }
}

void writeProjMapFile(string outputFile, vector<vector<Vec2d>> &positions, vector<vector<double>> &meanDistances, int glyphSize) {
    H5Easy::File file(outputFile, H5Easy::File::Overwrite);
    string v = "proj";
    H5Easy::dump(file, "projType", v);
    int size = positions[0].size();
    H5Easy::dump(file, "size", size);
    int layerCount = positions.size();
    H5Easy::dump(file, "layerCount", layerCount);
    for (int layer = 0; layer < layerCount; ++layer) {
        for (int i = 0; i < size; ++i) {
            vector<double> coord(2);
            coord[0] = positions[layer][i].x();
            coord[1] = positions[layer][i].y();
            H5Easy::dump(file, "layers/" + to_string(layer) + "/" + to_string(i), coord);
        }

        H5Easy::dump(file, "layers/" + to_string(layer) + "/meanDistances", meanDistances[layer]);
    }
}

void addAttributeToMapFile(string outputFile, string attributeName, vector<int> &attributeVector) {
    H5Easy::File file(outputFile, H5Easy::File::ReadWrite);
    H5Easy::dump(file, attributeName, attributeVector);
}

void addAttributeToMapFile(string outputFile, string attributeName, string s) {
    H5Easy::File file(outputFile, H5Easy::File::ReadWrite);
    H5Easy::dump(file, attributeName, s);
}

vector<int> readGT(string gtFile) {
    std::ifstream in(gtFile.c_str());
    char line[MAX_SIZE];
    int index = 0;
    vector<int> result(0);
    while (!in.eof()) {
        in.getline(line, MAX_SIZE);
        string lines(line);
        // vector<string> token;
        // tokenize(lines, token, ",");
        // if (token.size() >= 0) {
            int gt = atoi(line);
            // weak
            result.push_back(gt);
            ++index;
        // }
    }

    return result;
}

int readGridMapFile(string filename, vector<map<int, map<int, int>>> &grids, vector<vector<double>> &meanDistances) {
    H5Easy::File file(filename, H5Easy::File::ReadWrite);
    int size = H5Easy::load<int>(file, "size");
    int layerCount = H5Easy::load<int>(file, "layerCount");
    int gridWidth = ceil(sqrt(size));
    for (int i = 0; i < layerCount; ++i) {
        map<int, map<int, int>> grid;
        for (int x = 0; x < gridWidth; ++x) {
            vector<int> ycolumn = H5Easy::load<vector<int>>(file, "layers/" + to_string(i) + "/x/" + to_string(x));
            for (int y = 0; y < gridWidth; ++y) {
                grid[x][y] = ycolumn[y];
            }
        }

        grids.push_back(grid);
        meanDistances.push_back(H5Easy::load<vector<double>>(file, "layers/" + to_string(i) + "/meanDistances"));
    }

    return layerCount;
}

int readProjMapFile(string filename, vector<vector<Vec2d>> &positions, vector<vector<double>> &meanDistances) {
    H5Easy::File file(filename, H5Easy::File::ReadWrite);
    int size = H5Easy::load<int>(file, "size");
    int layerCount = H5Easy::load<int>(file, "layerCount");
    int gridWidth = ceil(sqrt(size));
    for (int layer = 0; layer < layerCount; ++layer) {
        vector<Vec2d> pos(size);
        for (int i = 0; i < size; ++i) {
            vector<double> coords = H5Easy::load<vector<double>>(file, "layers/" + to_string(layer) + "/" + to_string(i));
            Vec2d v(coords[0], coords[1]);
            pos[i] = v;
        }

        positions.push_back(pos);
        meanDistances.push_back(H5Easy::load<vector<double>>(file, "layers/" + to_string(layer) + "/meanDistances"));
    }

    return layerCount;
}

vector<Vec2d> readOrigMapFile(string filename) {
    H5Easy::File file(filename, H5Easy::File::ReadWrite);
    int size = H5Easy::load<int>(file, "size");
    vector<Vec2d> pos(size);
    for (int i = 0; i < size; ++i) {
        vector<double> coords = H5Easy::load<vector<double>>(file, to_string(i));
        Vec2d v(coords[0], coords[1]);
        pos[i] = v;
    }

    return pos;
}

vector<int> readGTH5(string filename) {
    H5Easy::File file(filename, H5Easy::File::ReadWrite);
    vector<int> gt = H5Easy::load<vector<int>>(file, "gt");
    return gt;
}

vector<double> computeMeanDistances(map<int, map<int, int>> &grid, int gridWidth, map<int, VectorND> &vectors) {
    vector<double> result(gridWidth * gridWidth);
    map<int, double> mapResult;
    for (int x = 0; x < gridWidth; ++x) {
        for (int y = 0; y < gridWidth; ++y) {
            double sum = 0;
            int factor = 0;
            VectorND v = vectors[grid[x][y]];
            if (x > 1 && y > 1) {
                sum += v.dist2(vectors[grid[x-1][y-1]]);
                ++factor;
            }
            if (y > 1) {
                sum += v.dist2(vectors[grid[x][y-1]]);
                ++factor;
            }
            if (x < gridWidth - 1 && y > 1) {
                sum += v.dist2(vectors[grid[x+1][y-1]]);
                ++factor;
            }
            if (x > 1) {
                sum += v.dist2(vectors[grid[x-1][y]]);
                ++factor;
            }
            if (x < gridWidth - 1) {
                sum += v.dist2(vectors[grid[x+1][y]]);
                ++factor;
            }
            if (x > 1 && y < gridWidth - 1) {
                sum += v.dist2(vectors[grid[x-1][y+1]]);
                ++factor;
            }
            if (y < gridWidth - 1) {
                sum += v.dist2(vectors[grid[x][y+1]]);
                ++factor;
            }
            if (x < gridWidth - 1 && y < gridWidth - 1) {
                sum += v.dist2(vectors[grid[x+1][y+1]]);
                ++factor;
            }

            mapResult[grid[x][y]] = sum / factor;
        }
    }

    for (int i = 0; i < gridWidth*gridWidth; ++i) {
        result[i] = mapResult[i];
    }

    return result;
}

Vec2d zero = Vec2d(0, 0);

vector<int> getNeighborhoodGrid(map<int, map<int, int>> &grid, int gridWidth, int x, int y) {
    vector<int> result;
    if (x > 1 && y > 1) {
        result.push_back(grid[x-1][y-1]);
    }
    if (y > 1) {
        result.push_back(grid[x][y-1]);
    }
    if (x < gridWidth - 1 && y > 1) {
        result.push_back(grid[x+1][y-1]);
    }
    if (x > 1) {
        result.push_back(grid[x-1][y]);
    }
    if (x < gridWidth - 1) {
        result.push_back(grid[x+1][y]);
    }
    if (x > 1 && y < gridWidth - 1) {
        result.push_back(grid[x-1][y+1]);
    }
    if (y < gridWidth - 1) {
        result.push_back(grid[x][y+1]);
    }
    if (x < gridWidth - 1 && y < gridWidth - 1) {
        result.push_back(grid[x+1][y+1]);
    }

    return result;
}

int sectorCount = 8;
vector<int> getNeighborhoodProj(vector<Vec2d> &positions, int posIndex, int sectorCount) {
    QuadTree qt;
    Vec2d minVec = positions[0];
    Vec2d maxVec = positions[0];
    int ss = 0;
    for (int i = 0; i < positions.size(); ++i) {
        if (!qt.isElement(positions[i])) {
            qt.addPoint(positions[i], i);
            minVec = wlp::minVector(minVec, positions[i]);
            maxVec = wlp::maxVector(maxVec, positions[i]);
        }
    }

    // cerr << minVec << " " << maxVec << endl;

    double maxRayX = (maxVec.x() - minVec.x()) / 2;
    double maxRayY = (maxVec.y() - minVec.y()) / 2;

    vector<int> sectors(sectorCount);
    Vec2d currentVector = positions[posIndex];
    Vec2d rayRect(.1, .1);
    for (int ii = 0; ii < sectorCount; ++ii) {
        sectors[ii] = -1;
    }

    int emptySector = sectorCount;
    while (emptySector > 0 && rayRect.x() < maxRayX && rayRect.y() < maxRayY) {
        rayRect *= 2;
        Rectd r(currentVector - rayRect, currentVector + rayRect);
        vector<uint> nearVectorIds;
        qt.getElements(r, nearVectorIds);
        for (int jj = 0; jj < nearVectorIds.size(); ++jj) {
            int j = nearVectorIds[jj];
            if (posIndex != j) {
                Vec2d vectorAngle = positions[j] - currentVector;
                float angle = atan2(vectorAngle[1], vectorAngle[0]);

                if (angle < 0.) angle += 2. * M_PI;
                int sector = int(floor(double(sectorCount) * angle / ( 2. * M_PI)));
                double dist = currentVector.dist(positions[j]);
                if (sectors[sector] == -1 || currentVector.dist(positions[sectors[sector]]) > dist) {
                    if (sectors[sector] == -1) {
                        --emptySector;
                    }

                    sectors[sector] = j;
                }
            }
        }
    }

    return sectors;

}
vector<double> computeMeanDistancesSectorsQT(vector<Vec2d> &positions, map<int, VectorND> & vectors) {
    vector<double> result(positions.size());
    QuadTree qt;
    Vec2d minVec = positions[0];
    Vec2d maxVec = positions[0];
    int ss = 0;
    for (int i = 0; i < positions.size(); ++i) {
        if (!qt.isElement(positions[i])) {
            qt.addPoint(positions[i], i);
            minVec = wlp::minVector(minVec, positions[i]);
            maxVec = wlp::maxVector(maxVec, positions[i]);
        }
    }

    // cerr << minVec << " " << maxVec << endl;

    double maxRayX = (maxVec.x() - minVec.x()) / 2;
    double maxRayY = (maxVec.y() - minVec.y()) / 2;

    for (int i = 0; i < positions.size(); ++i) {
        vector<int> sectors(sectorCount);
        Vec2d currentVector = positions[i];
        Vec2d rayRect(.1, .1);
        for (int ii = 0; ii < sectorCount; ++ii) {
            sectors[ii] = -1;
        }

        int emptySector = sectorCount;
        while (emptySector > 0 && rayRect.x() < maxRayX && rayRect.y() < maxRayY) {
            rayRect *= 2;
            Rectd r(currentVector - rayRect, currentVector + rayRect);
            vector<uint> nearVectorIds;
            qt.getElements(r, nearVectorIds);
            for (int jj = 0; jj < nearVectorIds.size(); ++jj) {
                int j = nearVectorIds[jj];
                if (i != j) {
                    Vec2d vectorAngle = positions[j] - currentVector;
                    float angle = atan2(vectorAngle[1], vectorAngle[0]);

                    if (angle < 0.) angle += 2. * M_PI;
                    int sector = int(floor(double(sectorCount) * angle / ( 2. * M_PI)));
                    double dist = currentVector.dist(positions[j]);
                    if (sectors[sector] == -1 || currentVector.dist(positions[sectors[sector]]) > dist) {
                        if (sectors[sector] == -1) {
                            --emptySector;
                        }

                        sectors[sector] = j;
                    }
                }
            }
        }

        double sum = 0;
        int factors = 0;
        for (int s = 0; s < sectorCount; ++s) {
            if (sectors[s] >= 0) {
                sum += vectors[i].dist2(vectors[sectors[s]]);
                factors += 1;
            }
        }

        result[i] = sum / factors;
    }

    return result;
}

// vector<double> computeMeanDistancesSectors(vector<Vec2d> &positions, map<int, VectorND> &vectors) {
//     vector<double> result(positions.size());

//     // safe yet slow N² distance computation
//     for (int i = 0; i < positions.size(); ++i) {
//         map<int, int> sectors;
//         Vec2d currentVector = positions[i];
//         for (int ii = 0; ii < 6; ++ii) {
//             sectors[ii] = -1;
//         }
//         for (int j = 0; j < positions.size(); ++j) {
//             if (i != j) {
//                 // cerr << positions[j] << endl;
//                 Vec2d vectorAngle = positions[j] - currentVector;
//                 // cerr << vectorAngle << endl;
//                 float angle = atan2(vectorAngle[1], vectorAngle[0]);

//                 if (angle < 0.) angle += 2. * M_PI;
//                 int sector = int(floor(double(6) * angle / ( 2. * M_PI)));
//                 double dist = positions[i].dist(positions[j]);
//                 if (sectors[sector] == -1 || positions[i].dist(positions[sectors[sector]]) > dist) {
//                     sectors[sector] = j;
//                 }
//             }
//         }

//         double sum = 0;
//         int factors = 0;
//         for (int s = 0; s < 6; ++s) {
//             if (sectors[s] >= 0) {
//                 sum += vectors[i].dist2(vectors[sectors[s]]);
//                 factors += 1;
//             }
//         }

//         result[i] = sum / factors;
//     }

//     return result;
// }

vector<double> computeMeanDistances(vector<Vec2d> &positions, map<int, VectorND> &vectors) {
    vector<double> result(positions.size());
    IncrementalDelaunay incr;
    vector<node> idToNode(positions.size());
    map<node, int> nodeToId;
    for (int i = 0; i < positions.size(); ++i) {
        node n = incr.addPoint(positions[i]);
        idToNode[i] = n;
        nodeToId[n] = i;
    }

    for (int i = 0; i < positions.size(); ++i) {
        vector<int> removed;
        removed.push_back(i);
        // cerr << "del " << positions[i] << endl;
        incr.delNode(idToNode[i]);
        for (int k = 0; k < 8; ++k) {
            node kn = incr.getClosestNode(positions[i]);
            int ki = nodeToId[kn];
            // cerr << "del " << positions[ki] << endl;
            removed.push_back(ki);
            incr.delNode(kn);
        }

        double sum = 0;
        VectorND v = vectors[i];
        // cerr << "add " << positions[i] << endl;
        node n = incr.addPoint(positions[i]);
        idToNode[i] = n;
        nodeToId[n] = i;
        for (int rki = 1; rki <= 8; ++rki) {
            int ki = removed[rki];
            sum += v.dist2(vectors[ki]);
            // cerr << "add " << positions[ki] << endl;
            node kn = incr.addPoint(positions[ki]);
            idToNode[ki] = kn;
            nodeToId[kn] = ki;
        }
        result[i] = sum / 8;
    }

    return result;
}

int getFileSizeArgument(string vectorFile) {
    H5Easy::File file(vectorFile, H5Easy::File::ReadWrite);
    int size = H5Easy::load<int>(file, "size");
    return size;
}

bool isGridType(string mapFile) {
    H5Easy::File file(mapFile, H5Easy::File::ReadWrite);
    string t = H5Easy::load<string>(file, "projType");
    return strcmp(t.c_str(), "grid") == 0;
}

map<int, map<int, int>> buildIdGridFromVectors(vector<vector<VectorND>> &vectorGrid) {
    map<int, map<int, int>> result;
    for (int x = 0; x < vectorGrid.size(); ++x) {
        for (int y = 0; y < vectorGrid[x].size(); ++y) {
            result[x][y] = vectorGrid[x][y].id;
        }
    }

    return result;
}

vector<vector<VectorND>> buildGridWithVectors(map<int, map<int, int>> &idGrid, map<int, VectorND> &vectors, int gridWidth) {
    vector<vector<VectorND>> result(gridWidth);
    for (int x = 0; x < gridWidth; ++x) {
        result[x] = vector<VectorND>(gridWidth);
        for (int y = 0; y < gridWidth; ++y) {
            result[x][y] = vectors[idGrid[x][y]];
        }
    }

    return result;
}

map<int, map<int, int>> applySSM(vector<vector<VectorND>> &vectors, bool sortVectors) {
    SSM2D ssm(vectors, sortVectors);
    for (int i = 0; i < 30; ++i) {
        ssm.apply();
    }

    return buildIdGridFromVectors(vectors);
}

// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

vector<Vec2d> callTsne(string filename) {
    convertRNVectorsToMatrix(filename, "python/tsneDataX.h5");
    string output = exec("python3 python/tsne.py");
    cerr << "done applying tsne, output:" << endl;
    vector<Vec2d> result;
    std::stringstream str;
    str.str(output);
    char line[MAX_SIZE];
    int index = -1;
    while (!str.eof()) {
        if (index >= 0) {
            str.getline(line, MAX_SIZE);
            string lines(line);
            cerr << lines << endl;
            vector<string> token;
            if (lines.size() > 1) {
                tokenize(lines, token, ",");
                if (token.size() >= 0) {
                    double x = atof(token[0].c_str());
                    double y = atof(token[1].c_str());
                    result.push_back(Vec2d(x, y));
                }
            }
        }
            ++index;
    }

    cerr << "done reading output" << endl;

    return result;

}

vector<Vec2d> normalizePositions(vector<Vec2d> &pos) {
    double maxValue = 0;
    double minValue = 0;
    wlp::Vec2d min = pos[0];
    wlp::Vec2d max = pos[0];
    for (int i = 1; i < pos.size(); ++i) {
        min = wlp::minVector(min, pos[i]);
        max = wlp::maxVector(max, pos[i]);
    }

    vector<Vec2d> result(pos.size());
    Vec2d size = max - min;
    double sizeMax = std::max(size.x(), size.y());
    for (int i = 0; i < pos.size(); ++i) {
        result[i] = (pos[i] - min) / sizeMax;
    }

    return result;
}

map<int, map<int, int>> callSSMRust(vector<vector<VectorND>> &vectors, string filename) {

    convertRNVectorsToMatrix(filename, "ssmDatatemp.h5");
    string output = exec("cargo run --manifest-path ./rust/isomatch/dimrec/Cargo.toml --release --bin project -- --h5_samples ssmDatatemp.h5 --h5_key datakey --labels dummylabels.csv --stdout ssm --width 100 --height 100");
    cerr << "done applying ssm rust, output:" << endl;
    //read result file
    map<int, map<int, int>> result;
    //convert to map<int, map<int,int >>

    std::stringstream str;
    str.str(output);
    char line[MAX_SIZE];
    int index = -1;
    while (!str.eof()) {
        if (index >= 0) {
            str.getline(line, MAX_SIZE);
            string lines(line);
            cerr << lines << endl;
            cerr << "parsing .." << endl;
            vector<string> token;
            if (lines.size() > 1) {
                tokenize(lines, token, ",");
                if (token.size() >= 0) {
                    int x = atoi(token[0].c_str());
                    int y = atoi(token[1].c_str());
                    result[x][y] = index - 1;
                }
            }
        }
            ++index;
    }

    cerr << "done reading output" << endl;

    return result;
}

map<int, map<int, int>> buildDefaultIdGrid(int gridWidth) {
    map<int, map<int, int>> result;
    int index = 0;
    for (int x = 0; x < gridWidth; ++x) {
        for (int y = 0; y < gridWidth; ++y) {
            result[x][y] = index;
            ++index;
        }
    }

    return result;
}

vector<double> standardizeDistances(vector<double> &meanValues) {
    double sum = 0;
    int size = meanValues.size();
    vector<double> standardizedValues(size);
    for (int i = 0; i < size; ++i) {
        sum += meanValues[i];
    }
    double mean = sum / size;
    // std = (sum((itr .- mean(itr)).^2) / (length(itr) - 1))
    double stdTop = 0;
    for (int i = 0; i < size; ++i) {
        double v = meanValues[i] - mean;
        stdTop += v * v;
    }
    double std = sqrt(stdTop / (size - 1));
    for (int i = 0; i < size; ++i) {
        standardizedValues[i] = (meanValues[i] - mean) / std;
    }

    return standardizedValues;
}

void convertTsneResultForSSM(vector<Vec2d> &tsnePos, map<int, VectorND> &mapPos) {
    for (int i = 0; i < tsnePos.size(); ++i) {
        VectorND vn(2);
        Vec2d v2 = tsnePos[i];
        vn[0] = v2.x();
        vn[1] = v2.y();
        vn.id = i;
        mapPos[i] = vn;
    }
}

Color lerpColor(Color c1, Color c2, float l) {
    if (l > 1) {
        l = 1;
    }

    if (l < 0) {
        l = 0;
    }
    int varR = c2.r() - c1.r();
    int varG = c2.g() - c1.g();
    int varB = c2.b() - c1.b();
    int vR = l * varR;
    int vG = l * varG;
    int vB = l * varB;
    return Color(c1.r() + vR, c1.g() + vG, c1.b() + vB);
}

Color getSet1ColorFromValue(int v) {
    Color arr[] = {
        Color(228, 26, 28),
        Color(55, 126, 184),
        Color(77, 175, 74),
        Color(152, 78, 163),
        Color(255, 127, 0),
        Color(255, 255, 51),
        Color(166, 86, 40),
        Color(247, 129, 191),
        Color(153, 153, 153)
    };

    if (v > 8)
        return Color(0,0,0);
    else
        return arr[v];
}

vector<Color> jetColormap{
    Color(0, 0, 131),
    Color(0, 60, 170),
    Color(5, 255, 255),
    Color(255, 255, 0),
    Color(250, 0, 0),
    Color(128, 0, 0)
};

vector<Color> YlOrRdColormap{
    Color(255, 255, 204),
    Color(255, 237, 160),
    Color(254, 217, 118),
    Color(254, 178, 76),
    Color(253, 141, 60),
    Color(252, 78, 42),
    Color(227, 26, 28),
    Color(189, 0, 38),
    Color(128, 0, 38)
};

vector<Color> BuRdColormap {
    Color(5,48,97),
    Color(33,102,172),
    Color(67,147,195),
    Color(146,197,222),
    Color(209,229,240),
    Color(247,247,247),
    Color(253,219,199),
    Color(244,165,130),
    Color(214,96,77),
    Color(178,24,43),
    Color(103,0,31)
};

vector<Color> plasmaColormap {
    Color(13,8,135),
    Color(84,2,163),
    Color(139,10,165),
    Color(185,50,137),
    Color(219,92,104),
    Color(244,136,73),
    Color(254,188,43),
    Color(240,249,33)
};

vector<Color> revPlasmaColormap {
    Color(240,249,33),
    Color(254,188,43),
    Color(244,136,73),
    Color(219,92,104),
    Color(185,50,137),
    Color(139,10,165),
    Color(84,2,163),
    Color(13,8,135)
};

vector<Color> revInfernoColormap {
    Color(252,255,164),
    Color(250,193,39),
    Color(245,125,21),
    Color(212,72,66),
    Color(159,42,99),
    Color(101,21,110),
    Color(40,11,84),
    Color(0,0,4)
};

vector<Color> viridisColormap {
    Color(68,1,84),
    Color(70,50,127),
    Color(54,92,141),
    Color(39,127,142),
    Color(31,161,135),
    Color(74,194,109),
    Color(159,218,58),
    Color(253,231,37)
};

vector<Color> revViridisColormap {
    Color(253,231,37),
    Color(159,218,58),
    Color(74,194,109),
    Color(31,161,135),
    Color(39,127,142),
    Color(54,92,141),
    Color(70,50,127),
    Color(68,1,84)
};

vector<Color> BkWhColormap {    // also the default colormap
    Color(0, 0, 0),
    Color(255, 255, 255)
};

Color getColorFromValue(double v, vector<Color> &colormap) {
    if (v > 1 || v < 0) {
        return Color(0, 0, 0);
    }

    v = max(v, 0.);
    double step = 1. / (colormap.size() - 1);
    int istep = 0;
    while (v > (istep * step)) {
        ++istep;
    }

    --istep;

    double vmin = istep * step;
    double vmax = (istep + 1) * step;
    Color c1 = colormap[istep];
    Color c2 = colormap[istep + 1];
    vmax = vmax - vmin;
    v = v - vmin;
    v = v / vmax;
    return lerpColor(c1, c2, v);
}

vector<double> getSortedPivots(vector<double> &dists, int resolution) {
    set<double> sortedSet(dists.begin(), dists.end());
    vector<double> result(resolution);
    // for (int i = 0; i < dists.size(); ++i) {
    // //     sortedSet.insert(dists[i]);
    //     cerr << dists[i] << endl;
    // }

    // for (int i = 0; i < sortedSet.size(); ++i) {
    //     cerr << *next(sortedSet.begin(), i) << endl;
    // }
    cerr << sortedSet.size() << endl;


    double partSize = sortedSet.size() * 1. / resolution;
    for (int i = 0; i < resolution - 1; ++i) {
        int nid = (int) floor(i * partSize);
        // if (i == resolution - 1) {
            // nid = dists.size() - 1;
        // }
        result[i] = *next(sortedSet.begin(), nid);
    }

    result[resolution - 1] = *next(sortedSet.end(), -1);

    return result;
}

double getSortedDistanceValue(vector<double> &sorted, double dist) {
    int p = 0;
    while (dist > sorted[p] && p < sorted.size()) {
        ++p;
    }

    if (p == sorted.size()) {
        return 1.1;
    }

    if (p == 0) {
        return 0;
    }

    double d1 = sorted[p - 1];
    double d2 = sorted[p];

    double v = (p * 1. + (dist - d1)/(d2 - d1)) / sorted.size();
    if (v > 1) {
        cerr << "above 1" << endl;
    }
    return v;
}

void drawGrid(map<int, map<int, int>> &grid, VectorGraph &graph, NodeProperty<int> &vectorIdProperty,
    vector<double> &standardizedValues, vector<Mark> &marks, map<int, Mark> &mapmark, vector<int> &gts,
    double x, double y, double pointSize, bool useGT, vector<Color> &colormap, int onlyGT) {
    int nid = 0;
    int nb_nodes = graph.numberOfNodes();
    int GRID_SIZE = ceil(sqrt(graph.numberOfNodes()));
    dbVis->maxConnectionSize(2);
    vector<double> sorted = getSortedPivots(standardizedValues, 100);

    // cerr << "echelle:" << endl;
    // for (int i = 0; i < sorted.size(); ++i) {
    //     cerr << sorted[i] << endl;
    // }
    // cerr << "---" << endl;

    double maxValue = sorted[sorted.size() - 1];
    for (int k = 0; k < GRID_SIZE; ++k) {
        for (int i = 0; i < GRID_SIZE; ++i)
        {
            int vid = grid[k][i];
            if (onlyGT == -1 || gts[vid] == onlyGT) {

                Mark mark = dbVis->addMark().position(x +  k * pointSize, y + i * pointSize, 0);
                if (nid < graph.numberOfNodes()) {
                    vectorIdProperty[nid] = vid;
                } else {
                    //not enough point to fill the grid add some dummy points
                    vectorIdProperty[nid] = -1;
                }

                // double dist = meanDistances[layer][vid];
                double stdDist = standardizedValues[vid];
                double val = getSortedDistanceValue(sorted, stdDist);
                // double val = stdDist/maxValue;
                Color col;
                if (useGT) {
                    //weak
                    // col = getJetColorFromValue(gts[vid]*1./10.);
                    // col = getSet1ColorFromValue(gts[vid]);
                        col = getColorFromValue(gts[vid]*1./10., jetColormap);
                } else {
                        col = getColorFromValue(val, colormap);
                        // int lum = max(min(255, (int)(255 - val * 255)), 0);
                        // col = Color(lum, lum, lum, 255);
                }

                // col = Color(0,0,0,255);

                // mark.size(Size(pointSize * 0.75, pointSize * 0.75, 0.)).color(col);
                mark.size(Size(pointSize*1.1, pointSize*1.1, 0.)).color(col);
                mark.borderWidth(0).borderColor(col);
                mark.shape(wlp::Shape::SQUARE);
                marks.push_back(mark);
                mapmark[nid] = mark;
                ++nid;
            }
        }
    }
}

void drawProj(vector<Vec2d> &positions, VectorGraph &graph, NodeProperty<int> &vectorIdProperty,
    vector<double> &standardizedValues, vector<Mark> &marks, map<int, Mark> &mapmark, vector<int> &gts,
    double x, double y, double pointSize, bool useGT, vector<Color> &colormap, int onlyGT) {
    int nid = 0;
    int nb_nodes = graph.numberOfNodes();
    int glyphSize = ceil(sqrt(graph.numberOfNodes()));
    dbVis->maxConnectionSize(2);

    vector<double> sorted = getSortedPivots(standardizedValues, 100);

    for (int i = 0; i < nb_nodes; ++i) {
        int nid = i;
        int vid = i;
            Vec2d v = positions[nid];
            Mark mark = dbVis->addMark().position(x + v.x() * glyphSize, y + v.y() * glyphSize, 0);
            if (nid < graph.numberOfNodes()) {
                vectorIdProperty[nid] = vid;
            } else {
                //not enough point to fill the grid add some dummy points
                vectorIdProperty[nid] = -1;
            }

            double stdDist = standardizedValues[vid];
            double val = getSortedDistanceValue(sorted, stdDist);
            Color col;
            if (useGT) {
                //weak
                //col = getJetColorFromValue(gts[vid]*1./10.);
                if (onlyGT == -1 || gts[vid] == onlyGT) {
                col = getSet1ColorFromValue(gts[vid]);
                } else {
                    col = Color(255,255,255,10);
                }
                // col = Color(0,0,0,255);
            } else {
                col = getColorFromValue(val, colormap);
                // int lum = max(min(255, (int)(255 - val * 255)), 0);
                // col = Color(lum, lum, lum, 255);
            }
            mark.size(Size(pointSize, pointSize, 0.)).color(col);
            mark.borderWidth(0).borderColor(col);
            mark.shape(wlp::Shape::SQUARE);
            marks.push_back(mark);
            mapmark[nid] = mark;
            ++nid;
    }
}


void drawOrigProj(vector<Vec2d> &positions, VectorGraph &graph, NodeProperty<int> &vectorIdProperty,
    vector<Mark> &marks, map<int, Mark> &mapmark, vector<int> &gts,
    double x, double y, double pointSize) {
    int nid = 0;
    int nb_nodes = graph.numberOfNodes();
    int glyphSize = ceil(sqrt(graph.numberOfNodes()));
    dbVis->maxConnectionSize(2);

    for (int i = 0; i < nb_nodes; ++i) {
        int nid = i;
        int vid = i;
        Vec2d v = positions[nid];
        Mark mark = dbVis->addMark().position(x + v.x() * glyphSize, y + v.y() * glyphSize, 0);
        if (nid < graph.numberOfNodes()) {
            vectorIdProperty[nid] = vid;
        } else {
            //not enough point to fill the grid add some dummy points
            vectorIdProperty[nid] = -1;
        }

        Color col;
        col = getSet1ColorFromValue(gts[vid]);
        mark.size(Size(pointSize, pointSize, 0.)).color(col);
        mark.borderWidth(0).borderColor(col);
        mark.shape(wlp::Shape::SQUARE);
        marks.push_back(mark);
        mapmark[nid] = mark;
        ++nid;
    }
}

// map<double, int> computeDistanceHistogram(vector<double> &dists, int resolution) {
//     double distMin = dists[0];
//     double distMax = dists[0];
//     map<double, int> result;

//     for (int i = 1; i < dists.size(); ++i) {
//         double d = dists[i];
//         distMin = min(d, distMin);
//         distMax = max(d, distMax);
//     }

//     double incr = (distMax - distMin) / resolution;
//     for (int i = 0; i < dists.size(); ++i) {
//         double d = dists[i];
//         int f = 1;
//         while (d > f * incr) {
//             ++f;
//         }

//         result[f*incr] += 1;
//     }

//     return result;
// }

struct RNPoint {
    VectorND position;
    double radius;
    double density;
    vector<RNPoint*> exclusions;
    int size;
};

double randomAround0() {
    return rand() * 1. / (RAND_MAX * 1.) * 2. - 1;
}

double randomBetween0And1() {
    return rand() * 1. / (RAND_MAX * 1.);
}

vector<RNPoint> generateRandomPointsIntoHypersphereSafe(RNPoint pivot, int count) {
    // cerr << "--- pivot : " << pivot.position << " " << pivot.radius << endl;
    int dimsCount = pivot.position.size();
    double radius = pivot.radius;
    double density = pivot.density;
    vector<RNPoint> result(count);
    for (int i = 0; i < count; ++i) {
        result[i] = RNPoint();
        VectorND v(dimsCount);
        for (int d = 0; d < dimsCount; ++d) {
            v[d] = randomAround0() * radius * (1 - density) + pivot.position[d];
        }

        bool skip = false;
        for (int ex = 0; !skip && ex < pivot.exclusions.size(); ++ex) {
            if (pivot.radius * (1 - density) > pivot.exclusions[ex]->radius * (1 - pivot.exclusions[ex]->density)) {
                skip = pivot.exclusions[ex]->position.dist2(v) <= pivot.exclusions[ex]->radius * (1 - pivot.exclusions[ex]->density);
            }
        }

        if (!skip && pivot.position.dist2(v) <= radius * (1 - density)) {
            result[i].position = v;
        } else {
            --i;
        }
    }

    return result;
}


// WARNING : Not uniform distribution of points
// random data within hypersphere in RN : https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
// Padding is here to ensure that points are not too close to sphere boundaries so they can serve as cluster pivots
// vector<VectorND> generateRandomPointsIntoHypersphere(VectorND pivot, double radius, int count, double padding = 0.) {
//     int dimsCount = pivot.size();
//     vector<VectorND> result(count);
//     for (int i = 0; i < count; ++i) {
//         VectorND v(dimsCount);
//         double ssum = 0;
//         for (int d = 0; d < dimsCount; ++d) {
//             v[d] = randomAround0();
//             ssum += v[d] * v[d];
//         }

//         double f = 1. / sqrt(ssum) * radius * (1-padding) * randomBetween0And1();
//         for (int d = 0; d < dimsCount; ++d) {
//             v[d] = v[d] * f + pivot[d];
//         }

//         assert(pivot.dist2(v) <= radius * (1-padding));
//         result[i] = v;
//     }

//     return result;
// }

// compute radius of hypersphere to other without overlap, then apply separationFactor. If there's only one sphere, maxRadius is applied without factor.
// maxRadius should be specified if hyperspheres are inside hyperspheres themselves to not cross boudaries during data generation (= padding * HSradius)
vector<double> computeRadiusOfHypersheres(vector<RNPoint> &points, double separationFactor, double maxRadius = 100) {
    int pointCount = points.size();
    vector<double> result(pointCount);
    if (pointCount == 1) {
        points[0].radius = maxRadius;
        result[0] = maxRadius;
    } else {
        for (int i = 0; i < pointCount; ++i) {
            VectorND currentSuperPoint = points[i].position;
            double minDistance = currentSuperPoint.dist2(points[(i+1)%pointCount].position);
            for (int j = 0; j < pointCount; ++j) {
                VectorND otherSuperPoint = points[j].position;
                if (i != j) {
                    double currentDistance = currentSuperPoint.dist2(otherSuperPoint);
                    minDistance = min(minDistance, currentDistance);
                }
            }

            // result[i] = min(minDistance * separationFactor, maxRadius);
            points[i].radius = minDistance / (2 * separationFactor);
            result[i] = minDistance / (2 * separationFactor);
        }
    }

    for (int i = 0; i < pointCount; ++i) {
        assert(result[i] > 0);
    }

    return result;
}


// vector<VectorND> generateDataSafe(
//     int dataSize, // total points count.
//     int spaceDims, // number of dimensions for Rn.
//     int superPointCount, // number of points with certain no overlap between them. Has clusters.
//     vector<int> clusterCount, // number of clusters for each superPoint, can overlap between a same superPoint.
//     vector<double> separationFactors, // Percentage of separation for clusters of each superPoint (0% = full overlap; 100% = no overlap, can be next to each other; 100+% = no overlap, increased separation between clusters)
//     vector<double> superPointDataShare, // proportion of data allowed for a superPoint and its clusters. Values are normalized.
//     vector<double> clusterSizeVariation, // size variation of clusters for each superPoint. 0 = no variation, <1 = has x% variation.
//     //vector<int> fakeData // number of bad classified points (change their label into a random another after processing)
//     vector<double> clusterDensities // factor of cluster radius. Higher value = densier cluster
// ) {


vector<RNPoint> generateDataSafe(GeneratorParameters &params) {

    // step 1 : define number of elements for each superPoint
    vector<int> superPointsSize(params.superPointCount);
    // normalize proportions
    double sum = 0;
    for (int i = 0; i < params.superPointCount; ++i) {
        assert(params.superPointDataShare[i] > 0);
        sum += params.superPointDataShare[i];
    }

    int remainingData = params.dataSize;
    for (int i = 0; i < params.superPointCount - 1; ++i) {
        int localSize = floor(params.superPointDataShare[i] / sum * params.dataSize);
        superPointsSize[i] = localSize;
        remainingData -= localSize;
        // cerr << "SP " << i << " size:" << localSize << ", remaining :" << remainingData << endl;
    }

    //also ensure last superPoint has the remaining points
    superPointsSize[params.superPointCount - 1] = remainingData;

    // step 2 : generate superPoints into RN
    RNPoint origin;
    origin.position = VectorND(params.spaceDims);
    for (int i = 0; i < params.spaceDims; ++i) {
        origin.position[i] = 0;
    }

    origin.radius = 100*params.spaceDims;

    double clusterPadding = 0.5;
    vector<RNPoint> superPoints = generateRandomPointsIntoHypersphereSafe(origin, params.superPointCount);
    //step 2.5 : assign sizes to RNPoints
    for (int i = 0; i < params.superPointCount; ++i) {
        superPoints[i].size = superPointsSize[i];
    }

    //step 3 : compute distances between superPoint to determine radius of hyperspheres
    vector<double> superPointRadius = computeRadiusOfHypersheres(superPoints, 4, 20);

    // step 4 : for each superPoint, generate pivot for each cluster
    vector<vector<RNPoint>> clusterPivots(params.superPointCount);
    for (int i = 0; i < params.superPointCount; ++i) {
        clusterPivots[i] = generateRandomPointsIntoHypersphereSafe(superPoints[i], params.clusterCount[i]);
    }

    // step 5 : for each superPoint, compute radius of each cluster
    vector<vector<double>> clusterRadius(params.superPointCount);
    for (int i = 0; i < params.superPointCount; ++i) {
        clusterRadius[i] = computeRadiusOfHypersheres(clusterPivots[i], params.separationFactors[i], superPoints[i].radius * clusterPadding);
    }

    // step 6 : assign size of each cluster
    vector<vector<int>> clusterSizes(params.superPointCount);
    for (int i = 0; i < params.superPointCount; ++i) {
        int clusterSize = (int) floor(superPoints[i].size * (1 - params.clusterSizeVariation[i]) / params.clusterCount[i]);
        int dataPool = superPoints[i].size - clusterSize * params.clusterCount[i];
        vector<int> localClusterSize(params.clusterCount[i]);
        for (int j = 0; j < params.clusterCount[i]; ++j) {
            int additional = 0;
            if (j == params.clusterCount[i] - 1 || dataPool == 1) {
                additional = dataPool;
                dataPool = 0;
            } else {
                if (dataPool > 1) {
                    additional = rand() % dataPool;
                    dataPool -= additional;
                }
            }

            localClusterSize[j] = clusterSize + additional;
            clusterPivots[i][j].size = clusterSize + additional;
            clusterPivots[i][j].density = 0;
        }

        clusterSizes[i] = localClusterSize;
    }

    // log de la structure sur cerr (distance entre clusters d'un même superPoint, avec son rayon)
    // for (int i = 0; i < params.superPointCount; ++i) {
    //     for (int j = 0; j < params.clusterCount[i]; ++j) {
    //         for (int k = j + 1; k < params.clusterCount[i]; ++k) {
    //             cerr << "SP" << i << ", distance C " << j << " (r:" << clusterRadius[i][j] << ", s:" << clusterSizes[i][j] << ")-" << k << "(r:" << clusterRadius[i][k] << ", s:" << clusterSizes[i][k] << "): " << clusterPivots[i][j].position.dist2(clusterPivots[i][k].position) << endl;
    //         }
    //     }
    // }

    // step 6.5 : assign exlusion references (bubbles)
    for (int i = 0; i < params.superPointCount; ++i) {
        for (int ii = 0; ii < params.clusterCount[i]; ++ii) {
            for (int ij = 0; ij < params.clusterCount[i]; ++ij) {
                if (params.clusterExclusions[i][ii][ij]) {
                    clusterPivots[i][ii].exclusions.push_back(&clusterPivots[i][ij]);
                }
            }
        }
    }

    // step 7 : generate points and assign label
    vector<RNPoint> result(0);
    int clusterId = 0;
    int cursor = 0;
    for (int i = 0; i < params.superPointCount; ++i) {
        for (int j = 0; j < params.clusterCount[i]; ++j) {
            // cerr << clusterDensities[clusterId] << endl;
            float density = 0;
            if (params.clusterDensities.find(clusterId) != params.clusterDensities.end()) {
                density = params.clusterDensities[clusterId];
                clusterPivots[i][j].density = params.clusterDensities[clusterId];
            }
            vector<RNPoint> cluster = generateRandomPointsIntoHypersphereSafe(clusterPivots[i][j], clusterSizes[i][j]);
            for (int ii = 0; ii < cluster.size(); ++ii) {
                cluster[ii].position.id = clusterId;
            }
            ++clusterId;
            result.insert(result.begin() + cursor, cluster.begin(), cluster.end());
            cursor += cluster.size();
        }
    }

    assert(cursor == params.dataSize);
    int labelCount = clusterId;
    // step 8 : apply missclassified and shuffle labels
    int numToMiss = (int) floor(params.missclassifiedRate * params.dataSize);
    for (int i = 0; i < numToMiss; ++i) {
        int itemToChange = rand() % result.size();
        int currentLabel = result[itemToChange].position.id;
        int newLabel = rand() % labelCount;
        while (newLabel == currentLabel) {
            newLabel = rand() % labelCount;
        }

        int oldLabel = result[itemToChange].position.id;
        result[itemToChange].position.id = newLabel;
        cerr << "[outliers] changed class of " << itemToChange << " from " << oldLabel << " to " << newLabel << endl;
    }

    //shuffle
    cerr << "--- label shuffling ---" << endl;
    vector<int> labels(labelCount);
    for (int i = 0; i < labelCount; ++i) {
        labels[i] = i;
    }
    int newLabel = 0;
    for (int i = 0; i < result.size(); ++i) {
        result[i].position.id = result[i].position.id + labelCount;
    }
    while (labels.size() > 0) {
        int randLabelIndex = rand() % labels.size();
        int indexToReplace = labels[randLabelIndex];
        labels.erase(labels.begin() + randLabelIndex);
        for (int i = 0; i < result.size(); ++i) {
            if (result[i].position.id == labelCount + indexToReplace) {
                result[i].position.id = newLabel;
            }
        }
        cerr << "switched " << indexToReplace << " to " << newLabel << endl;
        ++newLabel;
    }

    cerr << "--- pivots ---" << endl;
    for (int i = 0; i < superPoints.size(); ++i) {
        cerr << i <<": " << superPoints[i].position << " ; radius = " << superPoints[i].radius << endl;
    }
    cerr << "--- clusters ---" << endl;
    for (int i = 0; i < superPoints.size(); ++i) {
        for (int j = 0; j < params.clusterCount[i]; ++j) {
            RNPoint p = clusterPivots[i][j];
            cerr << i << ":" << j << " --> " << p.position << " ; size = " << p.size << " ; radius = " << p.radius << " ; density = " << p.density << endl;
            for (int k = i+1; k < params.clusterCount[i]; ++k) {
                cerr << "    <--> " << k << " : " << p.position.dist2(clusterPivots[i][k].position) << endl;
            }
        }
    }

    cerr << "--- end ---" << endl;


    return result;
}

void writeGeneratedDataFile(string outputFile, vector<VectorND> &data) {
    H5Easy::File file(outputFile, H5Easy::File::Overwrite);
    int size = data.size();
    vector<int> gts(size);
    H5Easy::dump(file, "size", size);
    for (int i = 0; i < size; ++i) {
        VectorND vnd = data[i];
        vector<double> v(vnd.data.size());
        for (int j = 0; j < v.size(); ++j) {
            v[j] = vnd[j];
        }

        H5Easy::dump(file, to_string(i), v);
        gts[i] = vnd.id;
    }

    H5Easy::dump(file, "gt", gts);
}

bool checkIfDataHasGT(string path) {
    H5Easy::File file(path, H5Easy::File::ReadWrite);
    return file.exist("gt");
}


void safeSegFaultMessage() {
    cerr << "Main code done, future errors can be ignored." << endl;
}

vector<int> parseIntList(string text) {
    vector<string> tokens;
    tokenize(text, tokens, ",");
    vector<int> result(tokens.size());
    for (int i = 0; i < tokens.size(); ++i) {
        result[i] = atoi(tokens[i].c_str());
    }

    return result;
}

vector<map<int, map<int, bool>>> generateEmptyExclusionList(int spCount, int sizeMax) {
    vector<map<int, map<int, bool>>> result(spCount);
    
    for (int sp = 0; sp < spCount; ++sp) {
        result[sp] = map<int, map<int, bool>>();
        for (int i = 0; i < sizeMax; ++i) {
            result[sp][i] = map<int, bool>();
            for (int j = 0; j < sizeMax; ++j) {
                result[sp][i][j] = false;
            }
        }
    }

    return result;
}

vector<map<int, map<int, bool>>> parseExclusionList(string text, int spCount, int sizeMax) {
    vector<string> tokens;
    tokenize(text, tokens, ",");
    vector<map<int, map<int, bool>>> result(spCount);
    
    for (int sp = 0; sp < spCount; ++sp) {
        result[sp] = map<int, map<int, bool>>();
        for (int i = 0; i < sizeMax; ++i) {
            result[sp][i] = map<int, bool>();
            for (int j = 0; j < sizeMax; ++j) {
                result[sp][i][j] = false;
            }
        }
    }
    for (int i = 0; i < tokens.size(); ++i) {
        vector<string> subTokens;
        tokenize(tokens[i].c_str(), subTokens, ":");
        int sp = atoi(subTokens[0].c_str());
        int a = atoi(subTokens[1].c_str());
        int b = atoi(subTokens[2].c_str());
        result[sp][a][b] = true;
        result[sp][b][a] = true;
    }

    return result;
}

map<int, double> parseOptionalFloatList(string text) {
    vector<string> tokens;
    tokenize(text, tokens, ",");
    map<int, double> result;
    for (int i = 0; i < tokens.size(); ++i) {
        vector<string> subTokens;
        tokenize(tokens[i].c_str(), subTokens, ":");
        int a = atoi(subTokens[0].c_str());
        float b = atof(subTokens[1].c_str());
        result[a] = b;
    }

    return result;
}

vector<double> parseFloatList(string text) {
    vector<string> tokens;
    tokenize(text, tokens, ",");
    vector<double> result(tokens.size());
    for (int i = 0; i < tokens.size(); ++i) {
        result[i] = atof(tokens[i].c_str());
    }

    return result;
}

void addOutliersOnExistingData(PostGenerationOutlierGenerator &params) {
    // check data nature (ssm or tsne) and class count
    bool isGridData = isGridType(params.inputFile);
    int gwidth = ceil(sqrt(getFileSizeArgument(params.inputFile)));
    int dataSize = gwidth * gwidth; // ???
    // open existing data
    map<int, map<int, int>> grid;
    vector<Vec2d> proj;
    vector<Vec2i> idToGridPos(dataSize);
    vector<int> gts(dataSize);
    gts = readGTH5(params.inputFile);
    int gtCount = 0;
    for (int i = 0; i < gts.size(); ++i) {
        gtCount = max(gtCount, gts[i]);
    }
    ++gtCount;
    vector<vector<double>> meanDistances; // osef
    if (isGridData) {
        vector<map<int, map<int, int>>> grids;
        readGridMapFile(params.inputFile, grids, meanDistances);
        grid = grids[0];
        for (int x= 0; x < gwidth; ++x) {
            for (int y = 0; y < gwidth; ++y) {
                int id = grid[x][y];
                idToGridPos[id] = Vec2i(x, y);
            }
        }
    } else {
        vector<vector<Vec2d>> positions;
        readProjMapFile(params.inputFile, positions, meanDistances);
        proj = positions[0];
    }

    // compute each group outlier count
    int exactOutlierCountCluster = rand() % gtCount;
    vector<int> outlierCounts(gtCount);
    // output expected exact outlier count to cluster
    outlierCounts[exactOutlierCountCluster] = params.missCount;
    cerr << "h" << endl;
    cerr << "exact is clu " << exactOutlierCountCluster << endl;

    for (int i = 0; i < gtCount; ++i) {
        if (i != exactOutlierCountCluster) {
            int count = params.missCount * (1 + (0.5 - randomBetween0And1()) * params.variationMax);
            outlierCounts[i] = count;
        }
    }

    // apply outliers ->
    for (int currentClass = 0; currentClass < gtCount; ++currentClass) {
        vector<int> ids;
        for (int i = 0; i < dataSize; ++i) {
            if (gts[i] == currentClass) {
                ids.push_back(i);
            }
        }

        for (int i = 0; i < outlierCounts[currentClass]; ++i) {
            int randId = rand()%ids.size();
            int id = ids[randId];
        // check neighborhood according to data nature
        // change to other class
            if (isGridData) {
                Vec2i vec = idToGridPos[id];
                vector<int> neigh = getNeighborhoodGrid(grid, gwidth, vec.x(), vec.y());
                bool safe = true;
                for (int j = 0; j < neigh.size(); ++j) {
                    safe = safe && (gts[neigh[j]] == currentClass);
                }
                if (!safe) {
                    --i;
                } else {
                    int newLabel = rand() % gtCount;
                    while (newLabel == currentClass) {
                        newLabel = rand() % gtCount;
                    }

                    gts[id] = newLabel;
                }
            } else {
                vector<int> neigh = getNeighborhoodProj(proj, id, 8);
                bool safe = true;
                for (int j = 0; j < neigh.size(); ++j) {
                    safe = safe && (gts[neigh[j]] == currentClass);
                }
                if (!safe) {
                    --i;
                } else {
                    int newLabel = rand() % gtCount;
                    while (newLabel == currentClass) {
                        newLabel = rand() % gtCount;
                    }

                    gts[id] = newLabel;
                }
            }
        }
    }

    // save new data
    if (isGridData) {
        vector<map<int, map<int, int>>> grids;
        grids.push_back(grid);

        writeGridMapFile(params.outputFile, grids, meanDistances, gwidth);
        addAttributeToMapFile(params.outputFile, "gt", gts);
    } else {
        vector<vector<Vec2d>> poss;
        poss.push_back(proj);
        writeProjMapFile(params.outputFile, poss, meanDistances, gwidth);
        addAttributeToMapFile(params.outputFile, "gt", gts);
    }
}

//==============================================================
int main(int argc, char *argv[])
{
    srand(time(NULL));
    cerr << "Usage : nnSsmDemo <activations.h5>" << endl;
    cerr << "Controls : ..." << endl;

    bool ssmCpp = false;
    bool ssmRust = false;
    bool tsneMode = false;
    bool tsneAndSsmMode = false;
    bool mapMode = false;
    bool projMode = false;
    bool doOutput = false;
    bool addAttribute = false;
    bool addAttributeH5 = false;
    bool useGT = false;
    bool updateMapFile = false;
    int onlyGT = -1;
    vector<string> vectorFiles;
    vector<Color> colormap = BkWhColormap;
    string outputFile;
    string defaultFile;
    int layerCount = -1;
    // load RN data and apply SSM (C++ implementation). Result is saved into map file to be loaded with --map argument
    // --ssm-cpp <number of h5 files> <file1> <file2> ... --output <map file>
    if (argc > 2 && strcmp(argv[1], "--ssm-cpp") == 0) {
        ssmCpp = true;
        layerCount = atoi(argv[2]);
        for (int i = 0; i < layerCount; ++i) {
            vectorFiles.push_back(argv[3 + i]);
            cerr << vectorFiles[i] << endl;
        }
        defaultFile = vectorFiles[0];
        if (strcmp(argv[3 + layerCount], "--output") == 0) {
            outputFile = argv[4 + layerCount];
            doOutput = true;
        }
    }

    // load RN data and apply SSM (Rust implementation). Result is saved into map file to be loaded with --map argument
    // --ssm-rust <number of h5 files> <file1> <file2> ... --output <map file>
    if (argc > 2 && strcmp(argv[1], "--ssm-rust") == 0) {
        ssmRust = true;
        layerCount = atoi(argv[2]);
        for (int i = 0; i < layerCount; ++i) {
            vectorFiles.push_back(argv[3 + i]);
            cerr << vectorFiles[i] << endl;
        }
        defaultFile = vectorFiles[0];
        if (strcmp(argv[3 + layerCount], "--output") == 0) {
            outputFile = argv[4 + layerCount];
            doOutput = true;
        }
    }

    // load RN data and apply t-SNE (Python implementation). Result is saved into map file to be loaded with --map argument
    if (argc > 2 && strcmp(argv[1], "--tsne") == 0) {
        tsneMode = true;
        layerCount = atoi(argv[2]);
        for (int i = 0; i < layerCount; ++i) {
            vectorFiles.push_back(argv[3 + i]);
            cerr << vectorFiles[i] << endl;
        }
        defaultFile = vectorFiles[0];
        if (strcmp(argv[3 + layerCount], "--output") == 0) {
            outputFile = argv[4 + layerCount];
            doOutput = true;
        }
    }

    // load RN data, apply t-SNE (Python) then apply SSM (C++) on resulting projections.
    if (argc > 2 && strcmp(argv[1], "--tsneAndSsm") == 0) {
        tsneAndSsmMode = true;
        layerCount = atoi(argv[2]);
        for (int i = 0; i < layerCount; ++i) {
            vectorFiles.push_back(argv[3 + i]);
            cerr << vectorFiles[i] << endl;
        }
        defaultFile = vectorFiles[0];
        if (strcmp(argv[3 + layerCount], "--output") == 0) {
            outputFile = argv[4 + layerCount];
            doOutput = true;
        }
    }

    // generate RN data randomly with parameters (based on ssmDemo from 02/17/22)
    // --generate <num of elements n for a nxn sized grid> <N of RN> <class count> <data separation> <fake labels> <variation taille>
    if (argc > 2 && strcmp(argv[1], "--generate") == 0) {
        // int dataSize = 0;
        // int spaceDims = 0;
        // int superPointCount = 0;
        // vector<int> clusterCount;
        // vector<double> separationFactors;
        // vector<double> superPointDataShare;
        // vector<double> clusterSizeVariation;
        // vector<int> fakeData = parseIntList(argv[9]);
        // vector<double> clusterDensities;
        GeneratorParameters params;
        params.missclassifiedRate = 0;
        int argPos = 2;
        bool hasExclusionList = false;
        while (argPos < argc) {
            if (strcmp(argv[argPos], "--size") == 0) {
                params.dataSize = atoi(argv[argPos + 1]) * atoi(argv[argPos + 1]);
            } else if (strcmp(argv[argPos], "--dims") == 0) {
                params.spaceDims = atoi(argv[argPos + 1]);
            } else if (strcmp(argv[argPos], "--sp") == 0) {
                params.superPointCount = atoi(argv[argPos + 1]);
            } else if (strcmp(argv[argPos], "--spSize") == 0) {
                params.clusterCount = parseIntList(argv[argPos + 1]);
                params.maxClusterCount = params.clusterCount[0];
                for (int i = 1; i < params.clusterCount.size(); ++i) {
                    params.maxClusterCount = max(params.maxClusterCount, params.clusterCount[i]);
                }
            } else if (strcmp(argv[argPos], "--cluSep") == 0) {
                params.separationFactors = parseFloatList(argv[argPos + 1]);
            }else if (strcmp(argv[argPos], "--spShare") == 0) {
                params.superPointDataShare = parseFloatList(argv[argPos + 1]);
            }else if (strcmp(argv[argPos], "--cluSizeVar") == 0) {
                params.clusterSizeVariation = parseFloatList(argv[argPos + 1]);
            }else if (strcmp(argv[argPos], "--cluDensity") == 0) {
                params.clusterDensities = parseOptionalFloatList(argv[argPos + 1]);
                // clusterDensities = parseFloatList(argv[argPos + 1]);
            }else if (strcmp(argv[argPos], "--cluExclusion") == 0) {
                hasExclusionList = true;
                params.clusterExclusions = parseExclusionList(argv[argPos + 1], params.superPointCount, params.maxClusterCount);
            }else if (strcmp(argv[argPos], "--missRate") == 0) {
                params.missclassifiedRate = atof(argv[argPos + 1]);
            } else if (strcmp(argv[argPos], "--output") == 0) {
                outputFile = argv[argPos + 1];
                doOutput = true;
            }
            
            argPos += 2;
        }

        if (!hasExclusionList) {
            params.clusterExclusions = generateEmptyExclusionList(params.superPointCount, params.maxClusterCount);
        }

        vector<RNPoint> RNPoints = generateDataSafe(params);
        vector<VectorND> data(RNPoints.size());
        for (int i = 0; i < RNPoints.size(); ++i) {
            data[i] = RNPoints[i].position;
        }
        cerr << "transfered" << endl;


        if (doOutput) {
            writeGeneratedDataFile(outputFile, data);
        }

        safeSegFaultMessage();
        return 0;
    }

    // load map file for navigation in NN.
    // --map <map file (.h5)>
    if (argc > 2 && strcmp(argv[1], "--map") == 0) {
        mapMode = true;
        defaultFile = argv[2];
        if (argc > 3 && strcmp(argv[3], "--jet") == 0) {
            colormap = jetColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--ylorrd") == 0) {
            colormap = YlOrRdColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--burd") == 0) {
            colormap = BuRdColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--plasma") == 0) {
            colormap = plasmaColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--revplasma") == 0) {
            colormap = revPlasmaColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--revinferno") == 0) {
            colormap = revInfernoColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--revviridis") == 0) {
            colormap = revViridisColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--viridis") == 0) {
            colormap = viridisColormap;
        }

        if (argc > 3 && strcmp(argv[3], "--gt") == 0) {
            useGT = true;
        }

        if (argc > 5 && strcmp(argv[4], "--only") == 0) {
            onlyGT = atoi(argv[5]);
        }
    }


    if (argc > 2 && strcmp(argv[1], "--proj") == 0) {
        projMode = true;
        defaultFile = argv[2];
    }

    // add gt attribute to output file.
    // --gt <gt file .csv> --output <map file>
    if (argc > 2 && strcmp(argv[1], "--gt") == 0) {
        defaultFile = argv[2];
        addAttribute = true;
        if (strcmp(argv[3], "--output") == 0) {
            outputFile = argv[4];
            doOutput = true;
        }
    }

        if (argc > 2 && strcmp(argv[1], "--gth5") == 0) {
        defaultFile = argv[2];
        addAttributeH5 = true;
        if (strcmp(argv[3], "--output") == 0) {
            outputFile = argv[4];
            doOutput = true;
        }
    }

    // update output file with additional attributes.
    // --update <map file>
    if (argc > 2 && strcmp(argv[1], "--update") == 0) {
        defaultFile = argv[2];
        outputFile = argv[2];
        updateMapFile = true;
    }

    if (argc > 2 && strcmp(argv[1], "--outliers") == 0) {
        PostGenerationOutlierGenerator params;
        params.missCount = 0;
        params.variationMin = 0;
        params.variationMax = 0;
        int argPos = 2;
        while (argPos < argc) {
            if (strcmp(argv[argPos], "--input") == 0) {
                params.inputFile = argv[argPos + 1];
            } else if (strcmp(argv[argPos], "--output") == 0) {
                params.outputFile = argv[argPos + 1];
            } else if (strcmp(argv[argPos], "--count") == 0) {
                params.missCount = atoi(argv[argPos + 1]);
            } else if (strcmp(argv[argPos], "--varMin") == 0) {
                params.variationMin = atof(argv[argPos + 1]);
            } else if (strcmp(argv[argPos], "--varMax") == 0) {
                params.variationMax = atof(argv[argPos + 1]);
            }

            argPos += 2;
        }

        addOutliersOnExistingData(params);
    }

#ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
    cout << "OpenMp activated : " << omp_get_num_procs() << " procs available" << endl;

#endif

    glutInit(&argc, argv);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    if ((glutCreateWindow("Wulip Glut Viewer")) == GL_FALSE)
    {
        cerr << "Unable to create a OpenGl Glut window" << endl;
        exit(EXIT_FAILURE);
    }
    /* Set up glut callback functions */
    glutIdleFunc(glut_idle_callback);
    glutReshapeFunc(glut_reshape_callback);
    glutDisplayFunc(glut_draw_callback);
    glutSpecialFunc(glut_special);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);

    initVis();

    if (ssmCpp) {
        int gridWidth = ceil(sqrt(getFileSizeArgument(defaultFile)));
        vector<map<int, map<int, int>>> results(layerCount);
        vector<vector<double>> meanDistances(layerCount);
        map<int, map<int, int>> grid = buildDefaultIdGrid(gridWidth);
        map<int, VectorND> vectors;
        for (int layer = 0; layer < layerCount; ++layer) {
            cerr << "Processing " << vectorFiles[layer] << "..." << endl;
            loadRNVectorsH5(vectorFiles[layer], vectors);
            vector<vector<VectorND>> vectorGrid = buildGridWithVectors(grid, vectors, gridWidth);
            cerr << "\t Loaded, applying SSM..." << endl;
            map<int, map<int, int>> newIdGrid = applySSM(vectorGrid, layer == 0);
            results[layer] = newIdGrid;
            grid = newIdGrid;
            cerr << "\t Done, computing distances..." << endl;
            vector<double> mDistances = computeMeanDistances(grid, gridWidth, vectors);
            meanDistances[layer] = mDistances;
            cerr << "\t Done." << endl;
        }

        if (doOutput) {
            cerr << "All layer processed, writing results..." << endl;
            writeGridMapFile(outputFile, results, meanDistances, gridWidth);
            cerr << "Done." << endl;
        }

        if (checkIfDataHasGT(defaultFile)) {
            vector<int> gtsData = readGTH5(defaultFile);
            addAttributeToMapFile(outputFile, "gt", gtsData);
        }

        safeSegFaultMessage();
    }

    if (ssmRust) {
        int gridWidth = ceil(sqrt(getFileSizeArgument(defaultFile)));
        vector<map<int, map<int, int>>> results(layerCount);
        vector<vector<double>> meanDistances(layerCount);
        map<int, map<int, int>> grid = buildDefaultIdGrid(gridWidth);
        map<int, VectorND> vectors;
        for (int layer = 0; layer < layerCount; ++layer) {
            cerr << "Processing " << vectorFiles[layer] << "..." << endl;
            loadRNVectorsH5(vectorFiles[layer], vectors);
            vector<vector<VectorND>> vectorGrid = buildGridWithVectors(grid, vectors, gridWidth);
            cerr << "\t Loaded, applying SSM..." << endl;
            map<int, map<int, int>> newIdGrid = callSSMRust(vectorGrid, vectorFiles[layer]);
            results[layer] = newIdGrid;
            grid = newIdGrid;
            cerr << "\t Done, computing distances..." << endl;
            vector<double> mDistances = computeMeanDistances(grid, gridWidth, vectors);
            meanDistances[layer] = mDistances;
            cerr << "\t Done." << endl;
        }

        if (doOutput) {
            cerr << "All layer processed, writing results..." << endl;
            writeGridMapFile(outputFile, results, meanDistances, gridWidth);
            cerr << "Done." << endl;
        }

        safeSegFaultMessage();
    }

    if (tsneMode) {
        int gridWidth = ceil(sqrt(getFileSizeArgument(defaultFile)));
        // vector<map<int, map<int, int>>> results(layerCount);
        vector<vector<Vec2d>> results(layerCount);
        vector<vector<double>> meanDistances(layerCount);
        // map<int, map<int, int>> grid = buildDefaultIdGrid(gridWidth);
        map<int, VectorND> vectors;
        for (int layer = 0; layer < layerCount; ++layer) {
            cerr << "Processing " << vectorFiles[layer] << "..." << endl;
            loadRNVectorsH5(vectorFiles[layer], vectors);
            cerr << "\t Loaded, applying tSNE..." << endl;
            vector<Vec2d> newPositions = callTsne(vectorFiles[layer]);
            // for (int i = 0; i < 10; ++i) {
            //     cerr << newPositions[i] << endl;
            // }
            newPositions = normalizePositions(newPositions);
            results[layer] = newPositions;
            cerr << "\t Done, computing distances..." << endl;
            vector<double> mDistances = computeMeanDistancesSectorsQT(newPositions, vectors);
            meanDistances[layer] = mDistances;
            cerr << "\t Done." << endl;
        }

        if (doOutput) {
            cerr << "All layer processed, writing results..." << endl;
            writeProjMapFile(outputFile, results, meanDistances, gridWidth);
            cerr << "Done." << endl;
        }

        if (checkIfDataHasGT(defaultFile)) {
            vector<int> gtsData = readGTH5(defaultFile);
            addAttributeToMapFile(outputFile, "gt", gtsData);
        }

        safeSegFaultMessage();
    }

        if (tsneAndSsmMode) {
        int gridWidth = ceil(sqrt(getFileSizeArgument(defaultFile)));
        vector<map<int, map<int, int>>> results(layerCount);
        vector<vector<double>> meanDistances(layerCount);
        map<int, map<int, int>> grid = buildDefaultIdGrid(gridWidth);
        map<int, VectorND> vectors;
        map<int, VectorND> vectors2D;
        for (int layer = 0; layer < layerCount; ++layer) {
            cerr << "Processing " << vectorFiles[layer] << "..." << endl;
            loadRNVectorsH5(vectorFiles[layer], vectors);
            cerr << "\t Loaded, applying tSNE..." << endl;
            vector<Vec2d> newPositions = callTsne(vectorFiles[layer]);
            convertTsneResultForSSM(newPositions, vectors2D);
            vector<vector<VectorND>> vectorGrid = buildGridWithVectors(grid, vectors2D, gridWidth);
            cerr << "\t Loaded, applying SSM..." << endl;
            map<int, map<int, int>> newIdGrid = applySSM(vectorGrid, layer == 0);
            results[layer] = newIdGrid;
            grid = newIdGrid;
            cerr << "\t Done, computing distances..." << endl;
            vector<double> mDistances = computeMeanDistances(grid, gridWidth, vectors);
            meanDistances[layer] = mDistances;
            cerr << "\t Done." << endl;
        }

        if (doOutput) {
            cerr << "All layer processed, writing results..." << endl;
            writeGridMapFile(outputFile, results, meanDistances, gridWidth);
            cerr << "Done." << endl;
        }

        safeSegFaultMessage();
    }

    if (addAttribute) {
        vector<int> gtsData = readGT(defaultFile);
        addAttributeToMapFile(outputFile, "gt", gtsData);
        cerr << "Done" << endl;

        safeSegFaultMessage();
    }

        if (addAttributeH5) {
        vector<int> gtsData = readGTH5(defaultFile);
        addAttributeToMapFile(outputFile, "gt", gtsData);
        cerr << "Done" << endl;

        safeSegFaultMessage();
    }

    if (updateMapFile) {
        addAttributeToMapFile(outputFile, "projType", "grid");
        cerr << "Done" << endl;

        safeSegFaultMessage();
    }

    if (mapMode) {
        int gridWidth = ceil(sqrt(getFileSizeArgument(defaultFile)));
        vector<map<int, map<int, int>>> grids;
        vector<vector<Vec2d>> positions;
        vector<vector<double>> meanDistances;
        if (isGridType(defaultFile)) {
            layerCount = readGridMapFile(defaultFile, grids, meanDistances);
        } else {
            layerCount = readProjMapFile(defaultFile, positions, meanDistances);
        }
        vector<int> gts(gridWidth * gridWidth);
        if (useGT) {
            gts = readGTH5(defaultFile);
        }
        vector<vector<double>> standardizedValuesVector(layerCount);
        vector<vector<Mark>> marks(layerCount);
        vector<map<int, Mark>> mapmark(layerCount);
        vector<VectorGraph> graphs(layerCount);
        int layer = 0;
        for (int i = 0; i < layerCount; ++i) {
            // standardizedValuesVector[i] = standardizeDistances(meanDistances[i]);
            standardizedValuesVector[i] = meanDistances[i];
        }

        vector<NodeProperty<int>> vectorIdProperties(layerCount);


        for (int layer = 0; layer < layerCount; ++layer) {

            graphs[layer].alloc(vectorIdProperties[layer]);
            graphs[layer].addNodes(gridWidth * gridWidth);

            graphs[layer].alloc(nodePos);
            graphs[layer].alloc(oriPos);

            srand(time(0));

            int GRID_SIZE = ceil(sqrt(graphs[layer].numberOfNodes()));
            if (isGridType(defaultFile)) {
                drawGrid(grids[layer], graphs[layer], vectorIdProperties[layer], standardizedValuesVector[layer], marks[layer], mapmark[layer],
                    gts, layer * (GRID_SIZE + 10) * 1, 0, 1, useGT, colormap, onlyGT);
            } else {
                drawProj(positions[layer], graphs[layer], vectorIdProperties[layer], standardizedValuesVector[layer], marks[layer], mapmark[layer],
                    gts, layer * (GRID_SIZE + 10) * 1, 0, 1, useGT, colormap, onlyGT);
                // CHANGE TSNE PROJ POINT SIZE HERE       --
            }
            wlp::Text t = dbVis->addText();
            t.text("Layer " + to_string(layer)).x((GRID_SIZE / 2) + (GRID_SIZE + 10) * 10 * layer).y((GRID_SIZE + 2) * 10).textColor(Color(0, 0, 0, 255));
            t.size(100).font(0).anchor(wlp::Anchor::CENTER);

        }

        dbVis->setHeatmapDistCoef(.002);
        dbVis->setHeatmapMax(100);
        dbVis->minMarkSize(1);
        // dbVis->maxMarkSize(2);
        dbVis->swap();
        dbVis->center();
        dbVis->getCamera().swap();
        glutPostRedisplay();

        glutMainLoop();
    }

        if (projMode) {
            int gridWidth = ceil(sqrt(getFileSizeArgument(defaultFile)));
            vector<Vec2d> positions = readOrigMapFile(defaultFile);
            vector<int> gts(gridWidth * gridWidth);
            gts = readGTH5(defaultFile);
            vector<Mark> marks;
            map<int, Mark> mapmark;
            VectorGraph graph;
            NodeProperty<int> vectorIdProperty;

            graph.alloc(vectorIdProperty);
            graph.addNodes(gridWidth * gridWidth);

            graph.alloc(nodePos);
            graph.alloc(oriPos);

            srand(time(0));

            int GRID_SIZE = ceil(sqrt(graph.numberOfNodes()));
            drawOrigProj(positions, graph, vectorIdProperty, marks, mapmark,
                    gts, 0, 0, .1);
            wlp::Text t = dbVis->addText();


            dbVis->setHeatmapDistCoef(.002);
            dbVis->setHeatmapMax(100);
            // dbVis->minMarkSize(2);
            dbVis->maxMarkSize(2);
            dbVis->swap();
            dbVis->center();
            dbVis->getCamera().swap();
            glutPostRedisplay();

            glutMainLoop();

        }
    return 0;
}
