# 接続直後にコマンドを入力すると"yes"を求められるので、それをこなした後に次のコマンドを用いる。
sshpass -p "root" scp cpp/* root@192.168.1.100:/home/root/Vitis-AI/examples/VART/yolov3
#KV260 で以下のコマンドを使用する
#yolov3
#ln -s /usr/share/vitis_ai_library/models/dpu_yolov3/dpu_yolov3.xmodel dpu_yolov3.xmodel
#ln -s /home/root/Vitis-AI/examples/Vitis-AI-Library/samples/yolov3/video video
