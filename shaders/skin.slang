struct SkinVertex {
    float3 position;
    float3 normal;
    float2 texCoords;
};

struct SkinJointIndices {
    uint4 indices;
};

struct SkinWeights {
    float4 value;
};

struct SkinPushConstants {
    uint vertexCount;
}

[[vk::binding(0, 0)]]
StructuredBuffer<SkinVertex> input;

[[vk::binding(1, 0)]]
RWStructuredBuffer<SkinVertex> outputs;

[[vk::binding(2, 0)]]
StructuredBuffer<float4x4> jointMatrixes;

[[vk::binding(3, 0)]]
StructuredBuffer<SkinJointIndices> jointIndices;

[[vk::binding(4, 0)]]
StructuredBuffer<SkinWeights> jointWeights;

[[vk::push_constant]]
SkinPushConstants pushConstants;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID) {
    uint vertexIndex = dispatchThreadID.x;

    if (vertexIndex >= pushConstants.vertexCount)
        return;

    SkinVertex vertex = input[vertexIndex];

    uint4 joints = jointIndices[vertexIndex].indices;
    float4 weights = jointWeights[vertexIndex].value;

    float4x4 skinMatrix = 
        jointMatrixes[joints.x] * weights.x +
        jointMatrixes[joints.y] * weights.y +
        jointMatrixes[joints.z] * weights.z +
        jointMatrixes[joints.w] * weights.w;

    SkinVertex output;

    output.position = mul(skinMatrix, float4(vertex.position, 1.0)).xyz;
    output.normal = mul(skinMatrix, float4(vertex.normal, 0.0)).xyz;
    output.texCoords = vertex.texCoords;

    outputs[vertexIndex] = output;
}