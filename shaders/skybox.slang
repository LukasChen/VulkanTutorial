struct UniformBuffer {
    float4x4 proj;
    float4x4 view;
};

struct VSInput {
    float3 inPos;
    float3 inNormal;
    float2 inTexCoord;
};

struct V2F {
    float4 pos : SV_POSITION;
    float2 texCoord : TEXCOORD0;
    float3 direction : TEXCOORD1;
};

[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;
[[vk::binding(0, 1)]]
Sampler2D texture;

static const float PI = 3.14159265358979323846;
static const float INF = 3.4028234663852886e38;

static const float EarthRadius = 6360e3;
static const float AtmosphereRadius = 6420e3;
static const float RayleighScaleHeight = 7994.0;
static const float MieScaleHeight = 1200.0;
static const float3 BetaRayleigh = float3(3.8e-6, 13.5e-6, 33.1e-6);
static const float3 BetaMie = float3(21.0e-6, 21.0e-6, 21.0e-6);
static const float MieAsymmetry = 0.76;
static const uint ViewSamples = 32;
static const uint LightSamples = 8;
static const float ViewerAltitude = 1.0;
static const float SunIntensity = 20.0;
static const float MieExtinctionMul = 1.1;

[shader("vertex")]
V2F vertMain(VSInput input) {
    V2F output;

    float4x4 viewNoTranslation = ubo.view;
    viewNoTranslation[0][3] = 0.0f;
    viewNoTranslation[1][3] = 0.0f;
    viewNoTranslation[2][3] = 0.0f;
    output.pos = mul(ubo.proj, mul(viewNoTranslation, float4(input.inPos, 1.0f)));
    output.pos = output.pos.xyww;
    output.texCoord = input.inTexCoord;
    output.direction = input.inPos;
    return output;
}

bool raySphereIntersect(float3 origin, float3 direction, float radius, out float t0, out float t1) {
    t0 = 0.0;
    t1 = 0.0;
    float b = dot(direction, origin);
    float c = dot(origin, origin) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant <= 0.0) {
        return false;
    }

    float root = sqrt(discriminant);
    t0 = -b - root;
    t1 = -b + root;
    if (t0 > t1) {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    return true;
}

float3 computeIncidentLight(float3 origin, float3 direction, float tMin, float tMax, float3 sunDirection) {
    float t0;
    float t1;
    if (!raySphereIntersect(origin, direction, AtmosphereRadius, t0, t1) || t1 < 0.0) {
        return float3(0.0);
    }

    if (t0 > tMin && t0 > 0.0) {
        tMin = t0;
    }
    if (t1 < tMax) {
        tMax = t1;
    }

    float segmentLength = (tMax - tMin) / float(ViewSamples);
    float tCurrent = tMin;
    float3 sumRayleigh = float3(0.0);
    float3 sumMie = float3(0.0);
    float opticalDepthRayleigh = 0.0;
    float opticalDepthMie = 0.0;

    float mu = dot(direction, sunDirection);
    float phaseRayleigh = 3.0 / (16.0 * PI) * (1.0 + mu * mu);
    float phaseMie =
        3.0 / (8.0 * PI) *
        ((1.0 - MieAsymmetry * MieAsymmetry) * (1.0 + mu * mu)) /
        ((2.0 + MieAsymmetry * MieAsymmetry) *
            pow(1.0 + MieAsymmetry * MieAsymmetry - 2.0 * MieAsymmetry * mu, 1.5));

    for (uint i = 0; i < ViewSamples; ++i) {
        float3 samplePosition = origin + (tCurrent + segmentLength * 0.5) * direction;
        float height = length(samplePosition) - EarthRadius;

        float hr = exp(-height / RayleighScaleHeight) * segmentLength;
        float hm = exp(-height / MieScaleHeight) * segmentLength;
        opticalDepthRayleigh += hr;
        opticalDepthMie += hm;

        float t0Light;
        float t1Light;
        raySphereIntersect(samplePosition, sunDirection, AtmosphereRadius, t0Light, t1Light);
        float segmentLengthLight = t1Light / float(LightSamples);
        float tCurrentLight = 0.0;
        float opticalDepthLightRayleigh = 0.0;
        float opticalDepthLightMie = 0.0;

        uint j;
        for (j = 0; j < LightSamples; ++j) {
            float3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5) * sunDirection;
            float heightLight = length(samplePositionLight) - EarthRadius;
            if (heightLight < 0.0) {
                break;
            }

            opticalDepthLightRayleigh += exp(-heightLight / RayleighScaleHeight) * segmentLengthLight;
            opticalDepthLightMie += exp(-heightLight / MieScaleHeight) * segmentLengthLight;
            tCurrentLight += segmentLengthLight;
        }

        if (j == LightSamples) {
            float3 tau =
                BetaRayleigh * (opticalDepthRayleigh + opticalDepthLightRayleigh) +
                BetaMie * MieExtinctionMul * (opticalDepthMie + opticalDepthLightMie);
            float3 attenuation = float3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
            sumRayleigh += attenuation * hr;
            sumMie += attenuation * hm;
        }

        tCurrent += segmentLength;
    }

    return (sumRayleigh * BetaRayleigh * phaseRayleigh + sumMie * BetaMie * phaseMie) * SunIntensity;
}

[shader("fragment")]
float4 fragMain(V2F input) : SV_Target {
    float3 direction = normalize(input.direction);
    float3 sunDirection = normalize(float3(0.0, 0.35, -1.0));
    float3 origin = float3(0.0, EarthRadius + ViewerAltitude, 0.0);

    float t0;
    float t1;
    float tMax = INF;
    if (raySphereIntersect(origin, direction, EarthRadius, t0, t1) && t0 > 0.0) {
        tMax = t0;
    }

    float3 color = computeIncidentLight(origin, direction, 0.0, tMax, sunDirection);
    return float4(color, 1.0);
}
