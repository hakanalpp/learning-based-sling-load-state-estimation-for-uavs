using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Bu itin doğurduğu IMU ile çalışıyor iki kere çalışmış oluyor yani hata burdan geliyortdu fyı
public class TrueState : ICommandable {
	Rigidbody rb;
	const uint type = 0;
	
	void Awake () {
		rb = gameObject.GetComponentInParent(typeof(Rigidbody)) as Rigidbody;
	}
	
	void FixedUpdate () {
        server.SendHeader(type, full_name, time_server.GetPhysicsTicks());
        server.SendData(rb.position);
        server.SendData(rb.rotation);
        server.SendData(rb.velocity);
        server.SendData(rb.angularVelocity);
	}
}
