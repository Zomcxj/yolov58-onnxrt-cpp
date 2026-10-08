#include "obb.h"

static int sig(double d) {
    return (d > eps) - (d < -eps);
}

struct Points {
    double x,y; 
    Points() {}
    Points(double x,double y):x(x),y(y) {}
    bool operator==(const Points &p) const {
        return sig(x - p.x) == 0 && sig(y - p.y) == 0;
    }
};

static double cross(Points o, Points a, Points b) {  //叉积
    return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

static double area(Points* ps, int n) {
    ps[n] = ps[0];
    double res = 0;
    for(int i = 0; i < n; i++) {
        res += ps[i].x * ps[i+1].y - ps[i].y * ps[i+1].x;
    }
    return res / 2.0;
}

static int lineCross(Points a, Points b, Points c, Points d, Points& p) {
    double s1,s2;
    s1 = cross(a, b, c);
    s2 = cross(a, b, d);

    if(sig(s1) == 0 && sig(s2) == 0) 
        return 2;
    if(sig(s2 - s1) == 0) 
        return 0;

    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 1;
}

static void polygon_cut(Points* p, int& n, Points a, Points b, Points* pp) {
//    static Point pp[maxn];
    int m = 0; p[n] = p[0];

    for(int i = 0; i < n; i++) {
        if(sig(cross(a, b, p[i])) > 0) 
            pp[m++] = p[i];
        if(sig(cross(a, b, p[i])) != sig(cross(a, b, p[i+1])))
            lineCross(a, b, p[i], p[i+1], pp[m++]);
    }
    n = 0;
    for(int i = 0; i < m; i++) {
        if(!i || !(pp[i] == pp[i-1]))
            p[n++] = pp[i];
    }

    while(n > 1 && p[n-1] == p[0])
        n--;
}

//--------------------------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
static double intersectArea2(Points a, Points b, Points c, Points d) {
    Points o(0, 0);
    int s1 = sig(cross(o, a, b));
    int s2 = sig(cross(o, c, d));

    if(s1 == 0 || s2 == 0) {return 0.0;}
    if(s1 == -1) {std::swap(a, b);}
    if(s2 == -1) {std::swap(c, d);}

    Points p[10] = {o, a, b};
    int n = 3;
    Points pp[maxn];
    polygon_cut(p, n, o, c, pp);
    polygon_cut(p, n, c, d, pp);
    polygon_cut(p, n, d, o, pp);
    double res = fabs(area(p, n));

    if(s1 * s2 == -1) {res =- res;}
    return res;
}

//求两多边形的交面积
static double intersectArea(Points* ps1, int n1, Points* ps2, int n2) {
    if(area(ps1, n1) < 0)
        std::reverse(ps1, ps1 + n1);
    if(area(ps2, n2) < 0)
        std::reverse(ps2, ps2 + n2);

    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    double res = 0;

    for(int i = 0; i < n1; i++) {
        for(int j = 0; j < n2; j++) {
            res += intersectArea2(ps1[i], ps1[i+1], ps2[j], ps2[j+1]);
        }
    }
    return res; //assume res is positive!
}


static double iou_poly(std::vector<double> p, std::vector<double> q) {
    Points ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    double inter_area = intersectArea(ps1, n1, ps2, n2);
    // double union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    // double iou = inter_area / union_area;

    // std::cout << "inter_area:" << inter_area << std::endl;
    // std::cout << "union_area:" << union_area << std::endl;
    // std::cout << "iou:" << iou << std::endl;

    // 交集和小框比较
    double min_area;
    if (fabs(area(ps1, n1)) < fabs(area(ps2, n2)))
        min_area = fabs(area(ps1, n1));
    else
        min_area = fabs(area(ps2, n2));
    double iou = inter_area / min_area;

    return iou;
}

// cut oriimg with overlap
std::vector<std::vector<int>> imgSplit(cv::Mat resImg, int cropSize) {

    std::vector<std::vector<int>> setxywh;

    int cols = resImg.cols;
    int rows = resImg.rows;

    int overlap = std::max(int(cropSize / 100 * 20), 200);  // =0 no overlap
    int cutSize = cropSize - overlap;

    int numW = (cols + cutSize - 1) / cutSize;
    int numH = (rows + cutSize - 1) / cutSize;

    for (int y = 0; y < numH; y++) {
        for (int x = 0; x < numW; x++) {
            int lx = x * cutSize;
            int ly = y * cutSize;
            int w = std::min(lx + cropSize, cols) - lx;
            int h = std::min(ly + cropSize, rows) - ly;

            std::vector<int> xywh;
            xywh.push_back(lx);
            xywh.push_back(ly);
            xywh.push_back(w);
            xywh.push_back(h);
            setxywh.push_back(xywh);
        }
    }
    
    return setxywh;
}


// del same idx
static void delRepeat(std::vector<YolovOutput>& output, std::vector<int> idx) {

    std::sort(idx.begin(), idx.end());
    idx.erase(std::unique(idx.begin(), idx.end()), idx.end());

    for (auto it = idx.rbegin(); it != idx.rend(); ++it) {
        output.erase(output.begin() + *it);
    }
}

// iou compare
void obboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf) {

    // map coordinate to oriimg
    for (int i = 0; i < result.size(); i++) {
        cv::Point2f p[4];
        result[i].rotatedBox.points(p);
        result[i].rotatedBox.center.x += lx;
        result[i].rotatedBox.center.y += ly;
        for (int l = 0; l < 4; l++) {
            p[l].x += lx;
            p[l].y += ly;
        }    
    }

    // ---------------with pad--------------- //
    if (postResult.size() == 0) {
        for (int i = 0; i < result.size(); i++) {
            if (result[i].confidence >= conf)
                postResult.push_back(result[i]);     
        }
        return;
    }

    std::vector<int> idx1;
    std::vector<int> idx2;

    for (int i = 0; i < result.size(); i++) {
        if (result[i].confidence < conf) {
            idx1.push_back(i);
            continue;
        }

        if ((result[i].rotatedBox.center.x - lx) < 500 || (result[i].rotatedBox.center.y - ly) < 500) {
            cv::Point2f p[4];
            result[i].rotatedBox.points(p);
            std::vector<double> P;
            for (int l = 0; l < 4; l++) {
                P.push_back(p[l].x);
                P.push_back(p[l].y);
            }

            for (int j = 0; j < postResult.size(); j++) {
                cv::Point2f q[4];
                postResult[j].rotatedBox.points(q);
                std::vector<double> Q;
                for (int l = 0; l < 4; l++) {
                    Q.push_back(q[l].x);
                    Q.push_back(q[l].y);
                }

                if (iou_poly(P, Q) > 0.2) {
                    if (result[i].rotatedBox.size.area() < postResult[j].rotatedBox.size.area())     
                        idx1.push_back(i);
                    else
                        idx2.push_back(j);
                }
            }

            delRepeat(postResult, idx2);
            idx2.clear();
        } 
    }

    delRepeat(result, idx1);  
    // ---------------with pad--------------- //

    for (int i = 0; i < result.size(); i++) {
        postResult.push_back(result[i]);
    }
}


static double iou_hbb(YolovOutput bbox1, YolovOutput bbox2) {

    int xmin = std::max(bbox1.box.x, bbox2.box.x);
    int ymin = std::max(bbox1.box.y, bbox2.box.y);
    int xmax = std::min(bbox1.box.x + bbox1.box.width, bbox2.box.x + bbox2.box.width);
    int ymax = std::min(bbox1.box.y + bbox1.box.height, bbox2.box.y + bbox2.box.height);

    int inter_area = std::max(0, xmax - xmin) * std::max(0, ymax - ymin);
    
    double iou;
    int area1 = bbox1.box.width * bbox1.box.height;
    int area2 = bbox2.box.width * bbox2.box.height;
    
    if (area1 < area2)
        iou = inter_area * 1.0 / area1;
    else
        iou = inter_area * 1.0 / area2;
       
    return iou; 
}


// iou compare
void hbboxFilter(std::vector<YolovOutput> result, std::vector<YolovOutput>& postResult, int lx, int ly, float conf) {

    // map coordinate to oriimg
    for (int i = 0; i < result.size(); i++) {
        result[i].box.x += lx;
        result[i].box.y += ly;  
    }

    // ---------------with pad--------------- //
    if (postResult.size() == 0) {
        for (int i = 0; i < result.size(); i++) {
            if (result[i].confidence >= conf)
                postResult.push_back(result[i]);     
        }
        return;
    }

    std::vector<int> idx1;
    std::vector<int> idx2;

    for (int i = 0; i < result.size(); i++) {
        if (result[i].confidence < conf) {     
            idx1.push_back(i);
            continue;
        }

        if ((result[i].box.x - lx) < 300 || (result[i].box.y - ly) < 300) {
            int iarea = result[i].box.width * result[i].box.height;

            for (int j = 0; j < postResult.size(); j++) {
                int jarea = postResult[j].box.width * postResult[j].box.height;
                
                if (iou_hbb(result[i], postResult[j]) > 0.7) {
                    if (iarea < jarea)     
                        idx1.push_back(i);
                    else
                        idx2.push_back(j);
                }
            }

            delRepeat(postResult, idx2);
            idx2.clear();
        }   
    }

    delRepeat(result, idx1);  
    // ---------------with pad--------------- //

    for (int i = 0; i < result.size(); i++) {
        postResult.push_back(result[i]);
    }
}
