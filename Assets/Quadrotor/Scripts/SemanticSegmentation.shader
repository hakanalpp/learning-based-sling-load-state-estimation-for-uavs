// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Custom/SemanticSegmentation"
{
    // See: https://forum.unity3d.com/threads/unlit-single-color-shader.180833/
	Properties
	{
		_SegmentationColor ("Segmentation Color", Color) = (0,1,0,1)
	}
	SubShader
	{
		Tags {"RenderType"="Opaque" }
		LOD 200
		Color[_SegmentationColor]
		Pass {}
	}
	SubShader
	{
		Tags {"RenderType"="Cutout" }
		LOD 200
		Color[_SegmentationColor]
		Pass {}
	}
	SubShader
	{
		Tags {"RenderType"="Fade" }
		LOD 200
		Color[_SegmentationColor]
		Pass {}
	}

	SubShader { 
    	Tags { "IgnoreProjector"="True" "RenderType"="Transparent" }
    	// LOD 200
    	// Color[_SegmentationColor]
		// Pass {}

		/////////////////////////////////////////////////////////
         /// First Pass
         /////////////////////////////////////////////////////////
 
         Pass {
             // Only render alpha channel
             ColorMask A
             Blend SrcAlpha OneMinusSrcAlpha
 
             CGPROGRAM
             #pragma vertex vert
             #pragma fragment frag
 
             fixed4 _SegmentationColor;
 
             float4 vert(float4 vertex : POSITION) : SV_POSITION {
                 return UnityObjectToClipPos(vertex);
             }
 
             fixed4 frag() : SV_Target {
                 return _SegmentationColor;
             }
 
             ENDCG
         }
 
         /////////////////////////////////////////////////////////
         /// Second Pass
         /////////////////////////////////////////////////////////
 
         Pass {
             // Now render color channel
             ColorMask RGB
             Blend SrcAlpha OneMinusSrcAlpha
 
             CGPROGRAM
             #pragma vertex vert
             #pragma fragment frag
 
             sampler2D _MainTex;
             sampler2D _SomeTex;
             fixed4 _SegmentationColor;
 
             struct appdata {
                 float4 vertex : POSITION;
                 float2 uv : TEXCOORD0;
             };
 
             struct v2f {
                 float2 uv : TEXCOORD0;
                 float4 vertex : SV_POSITION;
             };
 
             v2f vert(appdata v) {
                 v2f o;
                 o.vertex = UnityObjectToClipPos(v.vertex);
                 o.uv = v.uv;
                 return o;
             }
 
             fixed4 frag(v2f i) : SV_Target{
                 fixed4 col = _SegmentationColor;
                 return col;
             }
             ENDCG
         }


 	}

}