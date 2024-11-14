import datetime
import os
# import time

import PySimpleGUI as sg
import cv2
import numpy as np
import pandas as pd


# 初始化数据
folder_path = ''
df = None  # 初始化df
csv_file_path = None
object_name = ''
current_frame = 0
last_frame = 0
mission = 0
count = 0  # 每日任务数
frame_waiting = 150  # 每帧延迟100毫秒

def get_current_date():
    return datetime.datetime.now().strftime('%Y-%m-%d')

def read_mission_from_file(date):
    file_name = f"{date}-mission.txt"
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            return int(file.read())
    return 0

def write_mission_to_file(date, mission):
    file_name = f"{date}-mission.txt"
    with open(file_name, 'w') as file:
        file.write(str(mission))

def save_mission_to_file(mission):
    current_date = get_current_date()
    old_mission = read_mission_from_file(current_date)  # 如果日期不存在，则mission返回0
    mission += old_mission
    write_mission_to_file(current_date, mission)
    return mission
def save_to_csv_safely(dataframe, file_path,mission,isupdate):
    try:
        dataframe.to_csv(file_path, index=False)
        if not isupdate:
            # 每提交一次任务+1，更新操作则跳过
            mission += 1
            save_mission_to_file(mission)
        print(f"CSV saved successfully to {file_path}")

    except PermissionError:
        # current frame需要保持不变
        current_frame = int(values['-CUR-FRAME-'])
        window['-CUR-FRAME-'].update(f"{current_frame}")
        sg.popup(f"Error: CSV file {file_path} is open in Excel. Please close it and try again.")
        print(f"Error: CSV file {file_path} is open in Excel. Please close it and try again.")
    except Exception as e:
        # current frame需要保持不变
        current_frame = int(values['-CUR-FRAME-'])
        window['-CUR-FRAME-'].update(f"{current_frame}")
        sg.popup(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

upper_layout = [
    [sg.Checkbox('AU1', key='-AU1-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.Checkbox('AU2', key='-AU2-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.Checkbox('AU4', key='-AU4-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')),
     sg.Checkbox('AU5', key='-AU5-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.Checkbox('AU6', key='-AU6-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold'))],
    [sg.Checkbox('AU7', key='-AU7-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.Checkbox('AU43', key='-AU43-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold'))]
]



lower_layout = [
    [sg.Checkbox('AU' + str(col+8), size=(5, 1), key='-AU' + str(col+8) + '-', enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')) for col in range(3)],
    [[sg.Checkbox('AU' + str(row*5+col+12), size=(5, 1), key='-AU' + str(row*5+col+12) + '-', enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')) for col in range(5)] for row in range(3)],
    [[sg.Checkbox('AU' + str(row*5+col+28), size=(5, 1), key='-AU' + str(row*5+col+28) + '-', enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')) for col in range(5)] for row in range(2)]
]



# 定义窗口布局
layout = [
    [sg.Button('Open data path', key='-OPEN_PATH-',button_color='#2E8B57'), sg.InputText(key='-FILE_PATH-', size=(40, 1), disabled=True)],
    [sg.InputText(default_text='S1', key='-OBJECT_NAME-', size=(10, 1)), sg.Button('Confirm', key='-CONFIRM-',button_color='#2E8B57')],
    [sg.Text('', key='-STATUS-', size=(20, 1), text_color='#254336', font=('Arial', 20),background_color='#a8c1b4')],  # 这是状态条或文本元素
    [sg.Button('-', key='-SUB-',button_color='#2E8B57'), sg.InputText(default_text='0', key='-CUR-FRAME-', size=(10, 1)), sg.Button('+', key='-ADD-',button_color='	#2E8B57'),
     sg.Button('Open current frame', key='-OPEN-',button_color='#2E8B57'), sg.Button('Natural frame', key='-NATURAL-',button_color='#2E8B57')],
    [sg.Button('Play backwards 15 frames', key='-PLAYVIDEO_15-',button_color='#2E8B57'), sg.Button('Play backwards 30 frames', key='-PLAYVIDEO_30-',button_color='#2E8B57'), sg.Button('Play backwards 60 frames', key='-PLAYVIDEO_60-',button_color='#2E8B57'), sg.InputText(key='-FRAME_WAIT-', size=(10, 1), default_text='150')],
    # [sg.Frame(layout=upper_layout, title='Upper AU', title_color='red')],
    [sg.Text('Upper AU', size=(20, 1), font=('Arial', 15),background_color='#a8c1b4')],
    [upper_layout],
    [sg.Text('Lower AU/D', size=(20, 1), font=('Arial', 15),background_color='#a8c1b4')],
    [lower_layout],
    [sg.Checkbox('AU0', key='-AU0-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold'))],
    [sg.Checkbox('ASD', key='-ASD-', size=(5, 1), enable_events=True,background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold'))],
    [sg.Text('L:', size=(5, 1),background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.InputText(key='-L_INPUT-', size=(15, 1)), sg.Text('R:', size=(5, 1),background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.InputText(key='-R_INPUT-', size=(15, 1)), ],
    [sg.Text('T:', size=(5, 1),background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.InputText(key='-T_INPUT-', size=(15, 1)), sg.Text('B:', size=(5, 1),background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.InputText(key='-B_INPUT-', size=(15, 1)), ],
    [sg.Text(' ', size=(32, 1),background_color='#a8c1b4',text_color='#254336',font=('Helvetica', 12, 'bold')), sg.Button('Clear checkbox', key='-CLEAR-', size=(15, 1), font=('Arial', 15),button_color='#2E8B57')],
    [sg.Text('Current annotation frame num == picture frames num?', size=(50, 1), font=('Arial', 15, 'bold'), text_color='purple',background_color='#a8c1b4')],
    [sg.Text('', key='-FINAL_STATUS-', size=(50, 1), text_color='purple', font=('Arial', 15),background_color='#a8c1b4')],  # 最终评分提示
    [sg.Button('Submit', key='-SUBMIT-', size=(10, 1), font=('Arial', 15),button_color='#2E8B57')],
    [sg.Text('', key='-SUBMIT-STATUS-', size=(30, 1), text_color='red', font=('Arial', 20),background_color='#a8c1b4')],  # 更新提交状态
]

headers = ['frame', 'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU43', 'AU8', 'AU9', 'AU10', 'AU12', 'AU13',
                       'AU14', 'AU15', 'AU16', 'AU17', 'AU18', 'AD19', 'AU20', 'AU21', 'AU22', 'AU23', 'AU24', 'AU25', 'AU2X',
                       'AU28', 'AD29', 'AD30', 'AU31', 'AD32', 'AD33', 'AD34', 'AD35', 'AD36', 'AD37', 'AU0', 'ASD']
au_2_csv_idx = {}  # csv每列对应的索引
for idx, h in enumerate(headers):
    au_2_csv_idx[h] = idx


sg.theme('GreenMono')
# 创建窗口
window = sg.Window('AU Labeling', layout, resizable=True)

# 事件循环，持续捕获窗口的事件
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        # 写出当日任务到文件
        new_mission = save_mission_to_file(mission)
        sg.popup(f"恭喜您！今日标记完成{new_mission}帧！再接再厉！！！")
        break
    # 用户点击"打开数据路径"按钮
    elif event == '-OPEN_PATH-':
        folder_path = sg.popup_get_folder('choose folder')
        if folder_path:
            window['-FILE_PATH-'].update(folder_path)
    # 用户点击"确认"按钮
    elif event == '-CONFIRM-':
        object_name = values['-OBJECT_NAME-']
        csv_file_path = os.path.join(folder_path, f"{object_name}.csv")
        print(csv_file_path)
        # 检查CSV文件是否存在
        if not os.path.exists(csv_file_path):
            # 创建一个新的CSV文件

            df = pd.DataFrame(columns=headers)
            df.to_csv(csv_file_path, index=False)  # 不保存索引
            last_frame = 0
            sg.popup('文件不存在，已创建新文件。')
        else:
            # object_name = values['-OBJECT_NAME-']
            backup_path = os.path.join(folder_path, f"{object_name}-backup.csv")

            df = pd.read_csv(csv_file_path)
            # 备份上一次的csv，免得弄错哪个地方
            df.to_csv(backup_path, index=False)
            if not df.empty:
                last_frame = df.iloc[-1, 0]
                # sg.popup_timed(f"您上一次标记到了第{last_frame}帧",auto_close_duration=2)
                window['-STATUS-'].update(f"上次退出已标记到第{last_frame}帧")
                window['-CUR-FRAME-'].update(f"{last_frame+1}")
                current_frame = last_frame
            else:
                sg.popup('文件是空的。')
    elif event in ('-ADD-', '-SUB-'):
        try:
            # 获取当前输入框的值，并进行转换
            current_frame = int(values['-CUR-FRAME-'])
            # 根据事件进行加或减操作
            if event == '-ADD-':
                current_frame += 1
            else:
                current_frame = max(current_frame - 1, 0)
            # 更新输入框的值
            window['-CUR-FRAME-'].update(value=str(current_frame))
        except ValueError:
            sg.popup_error('请输入有效的数字!')
    elif event == '-OPEN-':
        img_path = os.path.join(folder_path + '/origin/' + object_name, f"{current_frame}_0.jpg")
        try:
            os.startfile(img_path)
        except FileNotFoundError as e:
            print('%d_0.jpg不存在！', {current_frame})
            print(e)
            # sg.popup_warning(f"{current_frame}_0.jpg不存在！")
    elif event == '-NATURAL-':
        img_path = os.path.join(folder_path + '/natural', f"{object_name}.jpg")
        try:
            os.startfile(img_path)
        except FileNotFoundError as e:
            # print('%d_0.jpg不存在！',{current_frame})
            sg.popup_annoying(f"{object_name}.jpg不存在！")
    elif event in ('-PLAYVIDEO_15-', '-PLAYVIDEO_30-', '-PLAYVIDEO_60-'):
        play_frame = 0
        if event == '-PLAYVIDEO_15-':
            play_frame = 15
        elif event == '-PLAYVIDEO_30-':
            play_frame = 30
        elif event == '-PLAYVIDEO_60-':
            play_frame = 60
        # 获取上层路径
        parent_path = os.path.dirname(folder_path)
        # print(parent_path)
        video_path = os.path.join(parent_path + '/My labeling/origin video/', f"{object_name}.mp4")
        # print(video_path)

        cap = cv2.VideoCapture(video_path)
        cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)
        start_f = current_frame
        end_f = current_frame + play_frame * 1
        # 设置开始帧为第300帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        frame_counter = start_f  # 计数器
        while cap.isOpened() and frame_counter <= end_f:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Video Playback', frame)
            frame_counter += 1
            # 100毫秒播放一帧
            if cv2.waitKey(int(values['-FRAME_WAIT-'])) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif event == '-SUBMIT-':
        # 计算所有被选中的sg.Checkbox的数量
        checkbox_values = {k: v for k, v in values.items() if '-AU' in k or '-ASD' in k}
        print(checkbox_values.values())
        # 先把frame写进去
        current_frame = int(values['-CUR-FRAME-'])
        # waiting_frame = current_frame + 1 # 如果执行修正，修正完则立即回到待标记的新帧
        label = [current_frame]
        label += [1 if v else 0 for v in checkbox_values.values()]

        # 判断是否添加L,R 使用,分割所有元素
        au_l_input = values['-L_INPUT-']
        au_r_input = values['-R_INPUT-']
        au_t_input = values['-T_INPUT-']
        au_b_input = values['-B_INPUT-']
        au_l_list = []  # [‘AU1’，‘AU4“...]
        au_r_list = []
        au_t_list = []  # [‘AU1’，‘AU4“...]
        au_b_list = []
        # 'AD29', 'AD30', 'AU31', 'AD32', 'AD33', 'AD34', 'AD35', 'AD36', 'AD37'
        if au_r_input != '' and (au_r_input is not None):
            au_r_str = au_r_input.split(',')  # ['1','3','4']
            au_r_list += ['AD' + i if 29 <= int(i) < 42 else 'AU' + i for i in au_r_str]
        if au_l_input != '' and (au_l_input is not None):
            au_l_str = au_l_input.split(',')  # ['1','3','4']
            # au_l_list += ['AU' + i for i in au_l_str]
            au_l_list += ['AD' + i if 29 <= int(i) < 42 else 'AU' + i for i in au_l_str]
        if au_t_input != '' and (au_t_input is not None):
            au_t_str = au_t_input.split(',')  # ['1','3','4']
            au_t_list += ['AD' + i if 29 <= int(i) < 42 else 'AU' + i for i in au_t_str]
        if au_b_input != '' and (au_b_input is not None):
            au_b_str = au_b_input.split(',')  # ['1','3','4']
            au_b_list += ['AD' + i if 29 <= int(i) < 42 else 'AU' + i for i in au_b_str]
        # 遍历单侧AU，修改label数组中的值为L或R
        csv_l_idx = [au_2_csv_idx[au] for au in au_l_list]  # au在csv中的索引
        for index in csv_l_idx:
            label[index] = 2  # L标签为2
        csv_r_idx = [au_2_csv_idx[au] for au in au_r_list]  # au在csv中的索引
        for index in csv_r_idx:
            label[index] = 3  # R为3

        csv_t_idx = [au_2_csv_idx[au] for au in au_t_list]  # au在csv中的索引
        for index in csv_t_idx:
            label[index] = 4  # T标签为4
        csv_b_idx = [au_2_csv_idx[au] for au in au_b_list]  # au在csv中的索引
        for index in csv_b_idx:
            label[index] = 5  # B为5


        print(label)
        new_row = pd.DataFrame(np.array([label], dtype=np.int32), columns=df.columns, index=None)

        # ！！！！重新读取csv，防止手工修改被覆盖（即读取上一次新保存的csv)！！！！
        df = pd.read_csv(csv_file_path)

        # 如果是修正操作（帧已存在），则直接覆盖
        # 还有一种情况就是表格中间插入新行（比如在70帧和100帧中间漏了个71帧），这里没考虑，而是直接放在了表格最后一行
        if current_frame in df['frame'].values:
            df.loc[df['frame'] == current_frame] = label
            print('更新成功')

            # 读最后一行
            last_frame = df.iloc[-1, 0]
            window['-CUR-FRAME-'].update(value=str(last_frame+1))
            window['-SUBMIT-STATUS-'].update(f"已标记第{current_frame}帧，待标记第{last_frame+1}帧")
            # 保存新的
            save_to_csv_safely(df, csv_file_path,mission,True)
            # df.to_csv(csv_file_path, index=False)
        else:
            # 新的frame
            df = pd.concat([df, new_row], ignore_index=True)
            # 更新帧
            window['-CUR-FRAME-'].update(value=str(current_frame + 1))
            # 更新状态
            window['-SUBMIT-STATUS-'].update(f"已标记第{current_frame}帧，待标记第{current_frame + 1}帧")

            # 保存新的，防止打开excel忘记关闭
            save_to_csv_safely(df,csv_file_path,mission,False)


    elif event == '-CLEAR-':
        # 更新checkbox,归0
        checkbox_keys = [k for k, v in values.items() if '-AU' in k]
        for key in checkbox_keys:
            window[key].update(value=False)
        # 清空任务文本,保留ASD属性

        window['-FINAL_STATUS-'].update('ASD' if values['-ASD-'] else '')
        # 清空LRBT
        window['-L_INPUT-'].update('')
        window['-R_INPUT-'].update('')
        window['-T_INPUT-'].update('')
        window['-B_INPUT-'].update('')
    elif '-AU' or '-ASD-' in event:
        checked_options = [window[key].Text for key in values if values[key]==True]
        window['-FINAL_STATUS-'].update('+'.join(checked_options))


window.close()
