"""
網路攝影機測試腳本 (WebCam Testing Script)
==========================================
目的：驗證網路攝影機是否能正常擷取影像，並進行基本功能測試
作者：Fly Eddie
日期：2025年11月
版本：2.0（完全除錯版本）

本程式提供以下功能：
1. 初始化並測試網路攝影機連接
2. 讀取並顯示攝影機技術參數
3. 擷取即時影像並進行顯示
4. 支援按鍵指令保存和退出
5. 提供詳細的錯誤診斷訊息
"""

# ============================================================================
# 第一部分：導入必要的模組
# ============================================================================

import cv2                          # OpenCV 電腦視覺庫，用於影像處理和攝影機操作
import sys                          # 系統模組，用於程式流控和退出代碼
from pathlib import Path            # 路徑操作模組，用於跨平台檔案路徑管理


# ============================================================================
# 第二部分：常數定義和全域設定
# ============================================================================

# 定義攝影機索引（0 表示預設的內建攝影機或第一個連接的攝影機）
CAMERA_INDEX = 0

# 定義輸出目錄（相對路徑，基於指令執行位置）
OUTPUT_DIR = Path("outputs/webcam_test")

# 定義影像品質參數（JPEG 壓縮品質，0-100，100 為最高品質）
IMAGE_QUALITY = 95

# 定義按鍵代碼常數（用於 cv2.waitKey 返回值的比較）
KEY_Q = ord('q')                   # 字符 'q' 的 ASCII 代碼，用於正常退出
KEY_S = ord('s')                   # 字符 's' 的 ASCII 代碼，用於保存影像
KEY_ESC = 27                       # ESC 鍵的代碼，用於快速退出

# 定義 OpenCV 窗口標題常數
WINDOW_TITLE = "WebCam Test - Press 'q' or 'ESC' to exit, 's' to save"


# ============================================================================
# 第三部分：主測試函數定義
# ============================================================================

def ensure_output_directory():
    """
    確保輸出目錄存在的輔助函數
    
    功能說明：
    - 檢查輸出目錄是否存在
    - 若不存在，則建立該目錄及所有父目錄
    - parents=True 表示建立中間目錄
    - exist_ok=True 表示目錄已存在時不拋出異常
    
    返回值：
    - Path 物件，表示輸出目錄的完整路徑
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def initialize_camera():
    """
    初始化網路攝影機的函數
    
    功能說明：
    - 建立 VideoCapture 物件來存取攝影機硬體
    - CAMERA_INDEX=0 表示系統預設攝影機
    - 若有多個攝影機，可更改 CAMERA_INDEX 的值（1, 2, 3...）
    
    返回值：
    - cv2.VideoCapture 物件（已連接但尚未驗證是否開啟成功）
    
    異常處理：
    - 不直接拋出異常，由呼叫者檢查 isOpened() 狀態
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    return cap


def verify_camera_opened(cap):
    """
    驗證攝影機是否成功開啟的函數
    
    參數：
    - cap: cv2.VideoCapture 物件
    
    功能說明：
    - 使用 isOpened() 方法檢查攝影機連接狀態
    - 該方法返回布林值（True 表示成功，False 表示失敗）
    
    返回值：
    - 布林值，True 表示攝影機已開啟，False 表示開啟失敗
    """
    return cap.isOpened()


def get_camera_properties(cap):
    """
    讀取攝影機屬性的函數
    
    參數：
    - cap: cv2.VideoCapture 物件
    
    功能說明：
    - cv2.CAP_PROP_FRAME_WIDTH: 取得目前設定的影像寬度（像素）
    - cv2.CAP_PROP_FRAME_HEIGHT: 取得目前設定的影像高度（像素）
    - cv2.CAP_PROP_FPS: 取得每秒影像幀數（Frames Per Second）
    - int() 將浮點數轉換為整數（寬度和高度必須是整數）
    
    返回值：
    - 字典，包含三個鍵：'width'、'height'、'fps'
    - 例如：{'width': 1920, 'height': 1080, 'fps': 30.0}
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    return {
        'width': frame_width,
        'height': frame_height,
        'fps': fps
    }


def capture_single_frame(cap):
    """
    從攝影機擷取單一影像幀的函數
    
    參數：
    - cap: cv2.VideoCapture 物件
    
    功能說明：
    - cap.read() 從攝影機讀取一幀影像
    - 返回兩個值：
      1. ret (布林值)：True 表示成功讀取，False 表示讀取失敗
      2. frame (numpy 陣列)：影像矩陣，尺寸為 (高度, 寬度, 3)
         - 最後的 3 代表三個顏色通道（B, G, R）
         - OpenCV 使用 BGR 顏色順序而不是 RGB
    
    返回值：
    - 元組 (ret, frame)，與 cap.read() 返回值相同
    """
    ret, frame = cap.read()
    return ret, frame


def save_image(frame, filename=None):
    """
    保存影像到磁碟的函數
    
    參數：
    - frame: numpy 陣列，代表一幀影像
    - filename: 可選參數，指定自訂檔案名稱
    
    功能說明：
    - cv2.imwrite(路徑, 影像) 將影像保存為檔案
    - 副檔名決定了保存格式（.jpg、.png、.bmp 等）
    - IMAGE_QUALITY 參數用於 JPEG 品質控制（cv2.IMWRITE_JPEG_QUALITY）
    - 若 imwrite 成功，返回 True；失敗返回 False
    
    返回值：
    - Path 物件，指向已保存影像的完整路徑
    """
    if filename is None:
        # 若未指定檔名，使用預設名稱
        filename = "test_frame.jpg"
    
    save_path = OUTPUT_DIR / filename
    
    # 使用 cv2.IMWRITE_JPEG_QUALITY 參數控制 JPEG 壓縮品質
    cv2.imwrite(
        str(save_path),                           # 轉換為字串路徑
        frame,                                    # 要保存的影像矩陣
        [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY]  # JPEG 品質參數
    )
    
    return save_path


def add_text_to_frame(frame, text, position, font_size=1, color=(0, 255, 0)):
    """
    在影像上繪製文字的輔助函數
    
    參數：
    - frame: numpy 陣列，待修改的影像
    - text: 字符串，要繪製的文字內容
    - position: 元組 (x, y)，文字左上角的像素坐標
    - font_size: 浮點數，字體大小縮放因子（預設為 1）
    - color: 元組 (B, G, R)，文字顏色（BGR 格式，預設為綠色）
    
    功能說明：
    - cv2.putText() 在影像上繪製文字
    - cv2.FONT_HERSHEY_SIMPLEX：預定義字體類型
    - font_size (字體大小)：影響文字高度
    - 2 (厚度)：文字的線條厚度（像素數）
    - 此函數會直接修改原始 frame 參數（傳遞為參考）
    """
    cv2.putText(
        frame,                          # 目標影像（將被直接修改）
        text,                           # 要繪製的文字
        position,                       # (x, y) 坐標
        cv2.FONT_HERSHEY_SIMPLEX,      # 字體選擇
        font_size,                      # 字體大小
        color,                          # BGR 顏色
        2                               # 文字厚度（像素）
    )
    return frame


def handle_key_input(key, frame_count):
    """
    處理使用者按鍵輸入的函數
    
    參數：
    - key: 整數，cv2.waitKey() 返回的按鍵代碼
    - frame_count: 整數，目前已擷取的幀數
    
    功能說明：
    - key & 0xFF: 位元運算遮罰，提取按鍵代碼的低 8 位
      （某些系統的 waitKey 返回值包含高位的狀態信息）
    - 比較按鍵代碼與預定義常數來判斷用戶操作
    - 返回一個動作識別字符
    
    返回值：
    - 'exit': 使用者要求退出（'q' 或 ESC 鍵）
    - 'save': 使用者要求保存影像（'s' 鍵）
    - 'continue': 無操作，繼續執行（其他按鍵或無按鍵）
    """
    key = key & 0xFF  # 提取低 8 位
    
    if key == KEY_Q or key == KEY_ESC:
        return 'exit'
    elif key == KEY_S:
        return 'save'
    else:
        return 'continue'


def display_startup_info():
    """
    顯示程式啟動訊息的函數
    
    功能說明：
    - 使用 print() 函數輸出信息到控制台
    - 使用 "=" * 60 建立視覺分隔線
    - 提供清晰的程式初始化確認
    """
    print("=" * 60)
    print("網路攝影機測試程式 (WebCam Test Program)")
    print("=" * 60)
    print()


def display_camera_info(props):
    """
    顯示攝影機技術參數的函數
    
    參數：
    - props: 字典，包含 'width'、'height'、'fps' 鍵
    
    功能說明：
    - 以可讀格式輸出攝影機參數
    - 使用 f-string（格式化字符串）進行動態值插入
    - 提供視覺層級結構（使用縮進和符號區分）
    """
    print("[攝影機信息]")
    print(f"  解析度：{props['width']} × {props['height']} 像素")
    print(f"  幀率：{props['fps']:.1f} FPS")
    print()


def display_control_instructions():
    """
    顯示按鍵控制說明的函數
    
    功能說明：
    - 向使用者提供按鍵操作指南
    - 清晰列舉每個按鍵的功能
    - 幫助使用者在執行時進行正確的交互
    """
    print("[按鍵控制]")
    print("  'q' 鍵或 ESC：退出程式")
    print("  's' 鍵：保存當前影像幀")
    print()


def main_loop(cap, props):
    """
    主迴圈函數 - 處理即時影像擷取和顯示的核心邏輯
    
    參數：
    - cap: cv2.VideoCapture 物件，已初始化的攝影機
    - props: 字典，攝影機技術參數
    
    功能說明：
    本函數是程式的核心部分，執行以下操作：
    
    1. 初始化幀計數器和保存計數器
    2. 進入無限迴圈以持續擷取影像
    3. 對每一幀進行以下處理：
       - 讀取影像
       - 在影像上繪製信息文字
       - 顯示影像在視窗中
       - 等待並處理使用者按鍵
       - 執行相應的動作（保存或退出）
    4. 在退出時釋放資源
    
    返回值：
    - 無 (None)
    """
    frame_count = 0           # 用於計數已擷取的總幀數
    saved_count = 0           # 用於計數已保存的影像數
    
    print("[即時擷取] 開始顯示攝影機畫面...")
    print()
    
    while True:
        # 從攝影機讀取下一幀影像
        ret, frame = cap.read()
        
        # 檢查是否成功讀取影像
        if not ret:
            print("✗ 錯誤：無法繼續從攝影機讀取影像")
            print("  可能的原因：攝影機連接中斷或硬體故障")
            break
        
        # 增加幀計數
        frame_count += 1
        
        # 在影像左上角顯示當前幀編號
        frame = add_text_to_frame(
            frame,
            f"Frame: {frame_count}",     # 文字內容
            (10, 30),                    # 位置：距離左邊 10 像素，距離上方 30 像素
            font_size=1,
            color=(0, 255, 0)            # 綠色
        )
        
        # 在影像上顯示解析度信息
        frame = add_text_to_frame(
            frame,
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}",  # frame.shape[1] 是寬，[0] 是高
            (10, 70),                    # 位置：距離左邊 10 像素，距離上方 70 像素
            font_size=0.7,
            color=(0, 255, 0)            # 綠色
        )
        
        # 在視窗中顯示處理後的影像
        cv2.imshow(WINDOW_TITLE, frame)
        
        # 等待使用者按鍵（參數 1 表示等待 1 毫秒）
        # 若沒有按鍵按下，返回 -1
        # 若有按鍵按下，返回該按鍵的 ASCII 代碼
        key = cv2.waitKey(1)
        
        # 若沒有按鍵按下（key == -1），繼續迴圈
        if key == -1:
            continue
        
        # 處理按鍵輸入
        action = handle_key_input(key, frame_count)
        
        if action == 'exit':
            # 使用者要求退出
            print("✓ 使用者按下退出鍵")
            break
        
        elif action == 'save':
            # 使用者要求保存影像
            saved_count += 1
            save_filename = f"saved_frame_{frame_count}.jpg"
            save_path = save_image(frame, save_filename)
            print(f"✓ 影像已保存 [{saved_count}]：{save_path}")
    
    # 結束迴圈後，輸出統計信息
    print()
    print(f"[統計信息]")
    print(f"  總共擷取幀數：{frame_count}")
    print(f"  已保存影像數：{saved_count}")
    print()


def cleanup_resources(cap):
    """
    清理和釋放資源的函數
    
    參數：
    - cap: cv2.VideoCapture 物件
    
    功能說明：
    - cap.release()：釋放攝影機資源，使其他應用程式可使用該攝影機
      （重要：不釋放會導致攝影機被程式獨占）
    - cv2.destroyAllWindows()：關閉所有 OpenCV 視窗
      （釋放視窗相關的系統資源）
    - 確保在程式結束時正確進行清理，避免資源洩漏
    """
    cap.release()              # 釋放攝影機硬體資源
    cv2.destroyAllWindows()    # 銷毀所有 OpenCV 視窗


def test_webcam():
    """
    主測試函數 - 協調所有測試步驟的高階邏輯
    
    功能說明：
    此函數是程式的入口點，執行以下步驟：
    1. 建立輸出目錄
    2. 顯示啟動訊息
    3. 初始化攝影機
    4. 驗證攝影機是否成功開啟
    5. 若開啟成功，讀取攝影機參數
    6. 顯示攝影機信息和控制說明
    7. 進行主要測試迴圈
    8. 清理資源
    
    返回值：
    - 布林值：True 表示測試成功完成，False 表示測試失敗
    
    異常處理：
    - 若任何步驟失敗，提供詳細的錯誤診斷訊息
    - 確保資源被正確釋放
    """
    
    # 步驟 1：確保輸出目錄存在
    ensure_output_directory()
    
    # 步驟 2：顯示啟動訊息
    display_startup_info()
    
    # 步驟 3：初始化攝影機
    print("[初始化] 正在連接網路攝影機...")
    cap = initialize_camera()
    
    # 步驟 4：驗證攝影機是否成功開啟
    if not verify_camera_opened(cap):
        print("✗ 致命錯誤：無法開啟網路攝影機")
        print()
        print("[錯誤診斷]")
        print("  可能的原因包括：")
        print("  1. 攝影機未正確連接到 USB 連接埠")
        print("  2. 攝影機驅動程式未安裝或版本過舊")
        print("  3. 另一個應用程式正在使用攝影機")
        print("     （如 Zoom、Teams、OBS 等）")
        print("  4. 作業系統隱私設定禁止應用程式存取攝影機")
        print()
        print("[解決步驟]")
        print("  1. 檢查 USB 連接（試試不同的連接埠）")
        print("  2. 更新 GPU 驅動程式和攝影機驅動程式")
        print("  3. 關閉所有其他使用攝影機的應用程式")
        print("  4. 檢查 Windows 設定 > 隱私與安全 > 相機")
        print()
        return False
    
    print("✓ 攝影機已成功連接")
    print()
    
    # 步驟 5：讀取攝影機參數
    print("[讀取參數] 正在獲取攝影機技術規格...")
    props = get_camera_properties(cap)
    
    # 步驟 6：顯示攝影機信息
    display_camera_info(props)
    
    # 步驟 7：顯示控制說明
    display_control_instructions()
    
    # 步驟 8：執行主測試迴圈
    try:
        main_loop(cap, props)
    except Exception as e:
        print(f"✗ 執行期間發生異常錯誤：{e}")
        import traceback
        traceback.print_exc()
    
    # 步驟 9：清理資源
    print("[清理] 正在釋放資源...")
    cleanup_resources(cap)
    
    print("✓ 測試完成，資源已釋放")
    print("=" * 60)
    print()
    
    return True


# ============================================================================
# 第四部分：程式進入點
# ============================================================================

if __name__ == "__main__":
    """
    程式進入點 - 條件式執行
    
    功能說明：
    - if __name__ == "__main__": 是 Python 的標準慣例
    - 此條件確保該代碼僅在檔案被直接執行時運行
    - 若此檔案被其他模組 import，此代碼不會執行
    - 這樣可以方便地將此檔案作為模組重複使用
    
    執行流程：
    1. 呼叫 test_webcam() 函數執行主邏輯
    2. 根據返回值決定程式退出代碼
    3. sys.exit(0)：表示程式正常退出
    4. sys.exit(1)：表示程式因為錯誤而退出
    
    異常捕捉：
    - 捕捉所有未預期的異常
    - 打印詳細的錯誤堆棧追蹤（traceback）
    - 確保程式以適當的退出代碼終止
    """
    
    try:
        # 執行主測試函數
        success = test_webcam()
        
        # 根據測試結果設定退出代碼
        # 0 = 成功，1 = 失敗（這是 Unix 傳統）
        sys.exit(0 if success else 1)
    
    except Exception as e:
        # 捕捉任何未預期的異常
        print()
        print("✗ 發生未預期的異常錯誤：")
        print(f"  {e}")
        print()
        print("[詳細錯誤追蹤]")
        
        # 打印完整的錯誤堆棧追蹤，便於除錯
        import traceback
        traceback.print_exc()
        
        print()
        sys.exit(1)  # 以錯誤代碼結束程式
