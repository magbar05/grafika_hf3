#include "framework.h"

// cs�cspont �rnyal�
const char * vertSource = R"(
	#version 330				
    precision highp float;

	layout(location = 0) in vec2 cP;	// 0. bemeneti regiszter

	void main() {
		gl_Position = vec4(cP.x, cP.y, 0, 1); 	// bemenet m�r normaliz�lt eszk�zkoordin�t�kban
	}
)";

// pixel �rnyal�
const char * fragSource = R"(
	#version 330
    precision highp float;

	uniform vec3 color;			// konstans sz�n
	out vec4 fragmentColor;		// pixel sz�n

	void main() {
		fragmentColor = vec4(color, 1); // RGB -> RGBA
	}
)";

/*kd = diff�z visszaver�d�si sz�n
  ks = Spekul�ris visszaver�d�si sz�n
  ka = k�rnyezeti visszaver�d�si sz�n*/

#include <glm/vec3.hpp>
#include <cmath>
const int winWidth = 600, winHeight = 600;
struct Material {
    enum class Type {
        DIFFUSE,
        DIFFUSE_SPECULAR,
        REFLECTIVE,
        REFRACTIVE
    };

    Type type;
    glm::vec3 ka; // K�rnyezeti visszaver�d�s
    glm::vec3 kd; // Diff�z visszaver�d�s
    glm::vec3 ks; // Spekul�ris visszaver�d�s
    float shininess;
    glm::vec3 reflectivity; // T�kr�z�d�s m�rt�ke (sz�nf�gg� lehet)
    glm::vec3 refractiveIndex; // T�r�smutat� (sz�nf�gg� lehet)
    glm::vec3 extinctionCoefficient; // Kiolt�si t�nyez� (sz�nf�gg� lehet)
    glm::vec3 transmissionColor; // �thalad� f�ny sz�ne/m�rt�ke

    // Konstruktor diff�z anyaghoz
    Material(glm::vec3 _kd) :
        type(Type::DIFFUSE), ka(_kd* (float)M_PI), kd(_kd), ks(glm::vec3(0.0f)),
        shininess(1.0f), reflectivity(glm::vec3(0.0f)), refractiveIndex(glm::vec3(1.0f)),
        extinctionCoefficient(glm::vec3(0.0f)), transmissionColor(glm::vec3(1.0f)) {
    }

    // Konstruktor diff�z-spekul�ris anyaghoz
    Material(glm::vec3 _kd, glm::vec3 _ks, float _shininess) :
        type(Type::DIFFUSE_SPECULAR), ka(_kd* (float)M_PI), kd(_kd), ks(_ks),
        shininess(_shininess), reflectivity(glm::vec3(0.0f)), refractiveIndex(glm::vec3(1.0f)),
        extinctionCoefficient(glm::vec3(0.0f)), transmissionColor(glm::vec3(1.0f)) {
    }

    // Konstruktor t�kr�z� anyaghoz (sz�nf�gg� t�kr�z�ssel)
    Material(glm::vec3 _reflectivity) :
        type(Type::REFLECTIVE), ka(glm::vec3(0.0f)), kd(glm::vec3(0.0f)), ks(glm::vec3(1.0f)),
        shininess(1.0f), reflectivity(_reflectivity), refractiveIndex(glm::vec3(1.0f)),
        extinctionCoefficient(glm::vec3(0.0f)), transmissionColor(glm::vec3(1.0f)) {
    }

    // Konstruktor t�r� anyaghoz (�lland� t�r�smutat�val)
    Material(float _refractiveIndex) :
        type(Type::REFRACTIVE), ka(glm::vec3(0.0f)), kd(glm::vec3(0.0f)), ks(glm::vec3(1.0f)),
        shininess(10.0f), reflectivity(glm::vec3(0.05f)), refractiveIndex(glm::vec3(_refractiveIndex)),
        extinctionCoefficient(glm::vec3(0.0f)), transmissionColor(glm::vec3(1.0f)) {
    }

    // Konstruktor t�r� anyaghoz (sz�nf�gg� t�r�smutat�val �s kiolt�ssal)
    Material(glm::vec3 _refractiveIndex, glm::vec3 _extinctionCoefficient) :
        type(Type::REFRACTIVE), ka(glm::vec3(0.0f)), kd(glm::vec3(0.0f)), ks(glm::vec3(1.0f)),
        shininess(10.0f), reflectivity(glm::vec3(0.05f)), refractiveIndex(_refractiveIndex),
        extinctionCoefficient(_extinctionCoefficient), transmissionColor(glm::vec3(1.0f)) {
    }

    // �ltal�nosabb konstruktor, ha minden param�tert meg akarunk adni
    Material(Type _type, glm::vec3 _ka, glm::vec3 _kd, glm::vec3 _ks, float _shininess,
        glm::vec3 _reflectivity, glm::vec3 _refractiveIndex, glm::vec3 _extinctionCoefficient,
        glm::vec3 _transmissionColor) :
        type(_type), ka(_ka), kd(_kd), ks(_ks), shininess(_shininess),
        reflectivity(_reflectivity), refractiveIndex(_refractiveIndex),
        extinctionCoefficient(_extinctionCoefficient), transmissionColor(_transmissionColor) {
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material* material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye; lookat = _lookat; fov = _fov;
        vec3 w = eye - lookat;
        float windowSize = length(w) * tanf(fov / 2);
        right = normalize(cross(vup, w)) * (float)windowSize * (float)winWidth / (float)winHeight;
        up = normalize(cross(w, right)) * windowSize;
    }

    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2 * (X + 0.5f) / winWidth - 1) + up * (2 * (Y + 0.5f) / winHeight - 1) - eye;
        return Ray(eye, dir);
    }

    void Animate(float dt) {
        vec3 d = eye - lookat;
        eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
        set(eye, lookat, up, fov);
    }
};

class Intersectable {
protected:
    Material* material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

class Plane : public Intersectable {
    vec3 normal;
    vec3 p;
public:
    Plane(Material* _material, const vec3& _normal, const vec3& _pointOnPlane) {
        normal = _normal; p = _pointOnPlane; material = _material;
    }
    Hit intersect(const Ray& ray) override {
        Hit res;
        float denominator = glm::dot(normal, ray.dir);
        if (abs(denominator) < 1e-6f) { // A sug�r p�rhuzamos a s�kkal
            return Hit();
        }

        float t = glm::dot(p - ray.start, normal) / denominator;
        if (t >= 0) {
            glm::vec3 hitPoint = ray.start + t * ray.dir;
            res.t = t;
            res.normal = normal;
            res.material = material;
            res.position = hitPoint;
            return res;
        }
        else {
            return Hit();
        }
    }


};

class Cylinder : public Intersectable {
    vec3 direction;
    vec3 startp;
    float height;
public:
    Cylinder(const vec3 d, const vec3 s, float h, Material* _material) {
        direction = d; startp = s; height = h; material = _material;
    }
    Hit intersect(const Ray& ray) override {
        glm::vec3 oc = ray.start - startp;;
        glm::vec3 a = ray.dir - glm::dot(ray.dir, direction) * direction;
        glm::vec3 b = oc - glm::dot(oc, direction) * direction;

        float A = glm::dot(a, a);
        float B = 2*glm::dot(a, b);
        float C = glm::dot(b, b);

        float discriminant = B*B-4*A*C;

        if (discriminant < 0) {
            return Hit();
        }

        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = (-B - sqrtDiscriminant) / 2*A;
        float t2 = (-B + sqrtDiscriminant) / 2*A;

        Hit hit1 = Hit();
        Hit hit2 = Hit();

        // Ellen�rizz�k az els� metsz�spontot
        if (t1 >= 0) {
            glm::vec3 p1 = ray.start + t1 * ray.start;
            float distAlongAxis1 = glm::dot(p1 - startp, direction);
            if (distAlongAxis1 >= 0 && distAlongAxis1 <= height) {
                glm::vec3 normal1 = glm::normalize(p1 - startp - distAlongAxis1 * direction);
                hit1.material = material;
                hit1.normal = normal1;
                hit1.position = p1;
                hit1.t = t1;
            }
        }

        // Ellen�rizz�k a m�sodik metsz�spontot
        if (t2 >= 0) {
            glm::vec3 p2 = ray.start + t2 * ray.dir;
            float distAlongAxis2 = glm::dot(p2 - startp, direction);
            if (distAlongAxis2 >= 0 && distAlongAxis2 <= height) {
                glm::vec3 normal2 = glm::normalize(p2 - startp - distAlongAxis2 * direction);
                hit2.material = material;
                hit2.normal = normal2;
                hit2.position = p2;
                hit2.t = t1;
            }
        }

        if (hit1.t > 0 && hit2.t > 0) {
            return (hit1.t < hit2.t) ? hit1 : hit2;
        }
        else if (hit1.t > 0) {
            return hit1;
        }
        else {
            return hit2;
        }

    }
};

class Cone : public Intersectable {
    vec3 axis;
    vec3 apex;
    float angle;
    float height;
public:
    Cone(vec3 d, vec3 p, float alf, float h, Material* _material) {
        axis = d; apex = p; angle = alf; material = _material; height = h;
    }
    Hit intersect(const Ray& ray) override {
        glm::vec3 oc = ray.start - apex;
        float cosAngleSquared = cos(angle) * cos(angle);

        float dDotA = glm::dot(ray.dir, axis);
        float oCDotA = glm::dot(oc, axis);
        float dDotD = glm::dot(ray.dir, ray.dir);
        float oCDotOC = glm::dot(oc, oc);
        float oCDotD = glm::dot(oc, ray.dir);

        float a = dDotD * cosAngleSquared - dDotA * dDotA;
        float b = 2.0f * (oCDotD * cosAngleSquared - oCDotA * dDotA);
        float c = oCDotOC * cosAngleSquared - oCDotA * oCDotA;

        float discriminant = b * b - 4.0f * a * c;

        if (discriminant < 0) {
            return Hit();
        }

        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = (-b - sqrtDiscriminant) / (2.0f * a);
        float t2 = (-b + sqrtDiscriminant) / (2.0f * a);

        Hit hit1 = Hit();
        Hit hit2 = Hit();

        // Ellen�rizz�k az els� metsz�spontot
        if (t1 >= 0) {
            glm::vec3 p1 = ray.start + t1 * ray.dir;
            float distAlongAxis1 = glm::dot(p1 - apex, axis);
            if (distAlongAxis1 >= 0 && distAlongAxis1 <= height) {
                glm::vec3 normal1 = glm::normalize(glm::cross(glm::cross(axis, p1 - apex), p1 - apex));
                if (glm::dot(normal1, ray.dir) > 0) normal1 = -normal1; // Biztos�tjuk, hogy a norm�l kifel� mutasson
                hit1.material = material;
                hit1.normal = normal1;
                hit1.position = p1;
                hit1.t = t1;
            }
        }

        // Ellen�rizz�k a m�sodik metsz�spontot
        if (t2 >= 0) {
            glm::vec3 p2 = ray.start + t2 * ray.dir;
            float distAlongAxis2 = glm::dot(p2 - apex, axis);
            if (distAlongAxis2 >= 0 && distAlongAxis2 <= height) {
                glm::vec3 normal2 = glm::normalize(glm::cross(glm::cross(axis, p2 - apex), p2 - apex));
                if (glm::dot(normal2, ray.dir) > 0) normal2 = -normal2; // Biztos�tjuk, hogy a norm�l kifel� mutasson
                hit2.material = material;
                hit2.normal = normal2;
                hit2.position = p2;
                hit2.t = t2;
            }
        }

        if (hit1.t > 0 && hit2.t > 0) {
            return (hit1.t < hit2.t) ? hit1 : hit2;
        }
        else if (hit1.t > 0) {
            return hit1;
        }
        else {
            return hit2;
        }

    }
};

const float Epsilon = 0.0001f;


struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};


class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light*> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * (float)M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.4f, 0.4f, 0.4f);
        vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

       
    }

    void render(std::vector<vec3>& image) {
        float timeStart = getElapsedTime();
        for (int Y = 0; Y < winHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < winWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * winWidth + X] = vec3(color.x, color.y, color.z);
            }
        }
        printf("Rendering time: %d milliseconds\r", (int)((getElapsedTime() - timeStart) * 1000));
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable* object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = -bestHit.normal;
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;

        vec3 outRadiance = hit.material->ka * La;
        for (Light* light : lights) {
            Ray shadowRay(hit.position + hit.normal * Epsilon, light->direction);
            float cosTheta = dot(hit.normal, light->direction);
            if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
                outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                vec3 halfway = normalize(-ray.dir + light->direction);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
            }
        }
        return outRadiance;
    }

    void Animate(float dt) { camera.Animate(dt / 10); }
};

class GreenTriangleApp : public glApp {
	Geometry<vec2>* triangle = nullptr;  // geometria
	GPUProgram* gpuProgram = nullptr;	 // cs�cspont �s pixel �rnyal�k
public:
	GreenTriangleApp() : glApp("Green triangle") { }

	// Inicializ�ci�, 
	void onInitialization() {
		triangle = new Geometry<vec2>;
		triangle->Vtx() = { vec2(-0.8f, -0.8f), vec2(-0.6f, 1.0f), vec2(0.8f, -0.2f) };
		triangle->updateGPU();
		gpuProgram = new GPUProgram(vertSource, fragSource);
	}

	// Ablak �jrarajzol�s
	void onDisplay() {
		glClearColor(0, 0, 0, 0);     // h�tt�r sz�n
		glClear(GL_COLOR_BUFFER_BIT); // rasztert�r t�rl�s
		glViewport(0, 0, winWidth, winHeight);
		triangle->Draw(gpuProgram, GL_TRIANGLES, vec3(0.0f, 1.0f, 0.0f));
	}
};

GreenTriangleApp app;

