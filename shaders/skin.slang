struct SkinVertex {
    // Match the 32-byte C++ Vertex layout. Using float3 members here makes
    // std430 give the structure a 48-byte array stride.
    float4 positionNormalX;
    float4 normalYZTexCoords;
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

    float3 position = vertex.positionNormalX.xyz;
    float3 normal = float3(
        vertex.positionNormalX.w,
        vertex.normalYZTexCoords.xy
    );
    float2 texCoords = vertex.normalYZTexCoords.zw;

    uint4 joints = jointIndices[vertexIndex].indices;
    float4 weights = jointWeights[vertexIndex].value;

    float4x4 skinMatrix = 
        jointMatrixes[joints.x] * weights.x +
        jointMatrixes[joints.y] * weights.y +
        jointMatrixes[joints.z] * weights.z +
        jointMatrixes[joints.w] * weights.w;

    SkinVertex output;

    float3 skinnedPosition = mul(skinMatrix, float4(position, 1.0)).xyz;
    float3 skinnedNormal = mul(skinMatrix, float4(normal, 0.0)).xyz;

    output.positionNormalX = float4(skinnedPosition, skinnedNormal.x);
    output.normalYZTexCoords = float4(skinnedNormal.yz, texCoords);

    outputs[vertexIndex] = output;
}
