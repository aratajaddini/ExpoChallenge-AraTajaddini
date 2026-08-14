import serial
import serial.tools.list_ports
import time




def connect(default_port='COM3', baudrate=9600):
   
    arduino = None
    status_msg = ""
  
    try:
        arduino = serial.Serial(port=default_port, baudrate=baudrate, timeout=1)
        time.sleep(2) 
        status_msg = f" Connected Successfully on {default_port}"
        print(f" {status_msg}")
        return arduino, status_msg
    except Exception as e:
        print(f"⚠️ {default_port} Not Found: {e}. Searching other ports...")

   
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        try:
            arduino = serial.Serial(port=p.device, baudrate=baudrate, timeout=1)
            time.sleep(2)
            status_msg = f" Auto-Connected on {p.device}"
            print(f" {status_msg}")
            return arduino, status_msg
        except Exception:
            continue

    status_msg = " Disconnected (No Arduino Port Found)"
    print(f" {status_msg}")
    return arduino, status_msg


def send_to_arduino(message, arduino):
  
    if arduino and arduino.is_open:
       
       arduino.write(f"{message}\n".encode('utf-8'))
       print("RECIVED MEASSAGE BY ARDUINO/ROBOT:" , message)
       
    else:
        print(f"⚠️ Arduino is not connected. Skipped command for {message}")
        return False