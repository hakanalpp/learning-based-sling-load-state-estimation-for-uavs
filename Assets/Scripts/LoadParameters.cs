using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class LoadParameters : MonoBehaviour {
	public string config_file;

	bool parameters_loaded = false;
	Dictionary<string, string> parameters;

	public Dictionary<string, string> GetParameters() {
		if (!parameters_loaded) {
			LoadParameterFile();
		}
		return parameters;
	}

	void LoadParameterFile() {
		parameters = new Dictionary<string, string> ();
		char[] separators = { ':' };
		foreach (string line in File.ReadAllLines (config_file)) {
			string[] split = line.Split(separators);
			if(split.Length == 2 && !split[0].Contains("#")) {
				parameters[split[0].Trim()] = split[1].Trim();
			}
		}
		parameters_loaded = true;
	}

	void Start () {

	}

	void Update () {
	
	}
}
