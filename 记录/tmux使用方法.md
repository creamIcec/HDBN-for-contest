# tmux使用方法

## 指令列表

`tmux`: 新建窗口(当不在任何窗口中时)

`tmux ls`: 列出当前所有正在运行的窗口

`tmux attach -t <窗口id>`: 进入编号为`<窗口id>`的窗口

`tmux detach`: 从一个窗口中退出



python main.py --config ./config/team/mixformer_V1_JM.yaml --device 0