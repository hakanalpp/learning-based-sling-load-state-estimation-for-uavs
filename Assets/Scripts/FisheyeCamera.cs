using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FisheyeCamera : RGBCamera {
	const int type = 4;
	RenderFisheye fisheye_script;
	
	public RenderCubemap cubemap_script;
    public Shader shader;
    public float alpha = 4.0f;
    public float chi = 0.0f;
    public float focalLength = 1.0f;
	private Material _material;


	// void Update () {
		// if(fisheye_script == null) {
			// fisheye_script = GetComponent<RenderFisheye>();
		// }
        // server.SendHeader(type, full_name, time_server.GetFrameTicks());
		// server.SendData(fisheye_script.alpha);
		// server.SendData(fisheye_script.chi);
		// server.SendData(fisheye_script.focalLength);
		// SendImage();
	// }
	
    private Material material {
        get {
            if (_material == null) {
                _material = new Material(shader);
                _material.hideFlags = HideFlags.HideAndDontSave;
            }
            return _material;
        }
    }


    private void OnDisable() {
        if (_material != null)
            DestroyImmediate(_material);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
		// if(fisheye_script == null) {
	        // fisheye_script = GetComponent<FisheyeCamera>();
	    // }
        if (shader != null) {
       	    material.SetTexture("_Cube", cubemap_script.cubemap);
       	    material.SetFloat("_Alpha", alpha);
       	    material.SetFloat("_Chi", chi);
       	    material.SetFloat("_FocalLength", focalLength);
            Graphics.Blit(source, destination, material);
        } else {
            Graphics.Blit(source, destination);
        }
	    if (send_image)
	    {
		    server.SendHeader(type, full_name, time_server.GetFrameTicks());
		    server.SendData(alpha);
		    server.SendData(chi);
		    server.SendData(focalLength);
			SendImage(destination);
	    }
    }
}
