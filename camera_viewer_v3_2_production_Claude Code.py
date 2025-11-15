# ============================================================================
# 程式名稱：OpticalInspect - C270 攝影機即時監控系統（工業級 v3.2）
# 
# 核心功能：
#   ✓ 實時影像擷取（支援 1280×720 最高解析度）
#   ✓ 智能亮度／對比／飽和度調整（Trackbar 控制）
#   ✓ 精確 FPS 計算與顯示
#   ✓ 實時日期時間戳記
#   ✓ 整齊排列的判讀資訊（黑色字體）
#   ✓ 500×500 預設視窗（可自由拉伸）
#   ✓ 完善的資源管理與錯誤處理
#
# 作者：Fly Eddie
# 版本：3.2（生產版）
# 日期：2025-11-15
# ============================================================================

# ============================================================================
# 第一部分：模組導入與依賴管理
# ============================================================================

import sys                          # 系統模組（環境變數、程式終止等）
import os                           # 作業系統模組（路徑檢測、環境變數等）
import cv2                          # OpenCV 電腦視覺庫（影像處理的核心）
import time                         # 時間模組（FPS 計算、延遲等）
import numpy as np                  # NumPy 數值計算庫（陣列操作的基礎）
from datetime import datetime       # 日期時間模組（取得當前時間戳記）
from PIL import Image, ImageDraw, ImageFont  # PIL 圖像庫（中文文字渲染）


# ============================================================================
# 第二部分：全域常數定義（工業級系統參數）
# ============================================================================

# —— OpenCV 底層攝影機設定 ——
CAM_INDEX = 0                       # 攝影機索引編號（0=第一台、1=第二台等）
TARGET_WIDTH = 1280                 # 目標幀寬度（像素）- C270 最大值
TARGET_HEIGHT = 720                 # 目標幀高度（像素）- 720p 標準

# —— 視窗與介面設定 ——
WINDOW_NAME = "OpticalInspect - C270 Real-time Monitoring"  # 窗口英文標題
WINDOW_MODE = cv2.WINDOW_NORMAL     # 窗口類型（可調整大小）
WINDOW_WIDTH_INIT = 500             # 預設窗口寬度（像素）
WINDOW_HEIGHT_INIT = 500            # 預設窗口高度（像素）

# —— 影像調整初始參數 ——
BRIGHTNESS_INIT = 0                 # 亮度初始值（-100~+100，0=正常）
CONTRAST_INIT = 1.0                 # 對比度初始值（0.5~3.0，1.0=正常）
SATURATION_INIT = 1.0               # 飽和度初始值（0.0~2.0，1.0=正常）

# —— 文字顯示顏色定義（BGR 格式）——
COLOR_BLACK = (0, 0, 0)             # 黑色：判讀資訊顏色
COLOR_WHITE = (255, 255, 255)       # 白色：備用顏色
COLOR_CYAN = (255, 255, 0)          # 青色：備用顏色

# —— 文字位置常數 ——
INFO_START_X = 15                   # 判讀資訊起始 X 座標（離左邊 15px）
INFO_START_Y = 35                   # 判讀資訊起始 Y 座標（離上方 35px）
INFO_LINE_SPACING = 28              # 每行文字間距（28px）

# —— 中文字體設定 ——
try:
    # 嘗試第一優先：微軟雅黑（清晰度最佳）
    FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
    # 檢查該路徑是否存在
    if not os.path.exists(FONT_PATH):
        # 嘗試第二優先：宋體（備選方案）
        FONT_PATH = r"C:\Windows\Fonts\simsun.ttc"
    # 再次確認檔案存在
    if not os.path.exists(FONT_PATH):
        # 若無可用中文字體，設為 None
        FONT_PATH = None
    
    # 如果成功找到字體檔案
    if FONT_PATH:
        # 建立大號字體物件（字型大小=18pt）
        FONT = ImageFont.truetype(FONT_PATH, 18)
        # 建立小號字體物件（字型大小=14pt）
        FONT_SMALL = ImageFont.truetype(FONT_PATH, 14)
        # 中文支援標誌設為真
        CHINESE_SUPPORT = True
    else:
        # 無法找到字體檔案，禁用中文支援
        FONT = None
        FONT_SMALL = None
        CHINESE_SUPPORT = False
        
except Exception as e:
    # 若發生任何例外（如字體檔案損壞），禁用中文支援
    print(f"[WARNING] Chinese font initialization failed: {e}")
    FONT = None
    FONT_SMALL = None
    CHINESE_SUPPORT = False


# ============================================================================
# 第三部分：中文文字渲染函式
# ============================================================================

def put_text_chinese(frame, text, position, font_size=18, color=(0, 0, 0)):
    """
    在影像上繪製中文文字（支援中英文混合）
    
    參數說明：
        frame：輸入的影像陣列（numpy array，BGR 色彩空間）
        text：要繪製的文字內容（字串）
        position：文字位置的座標 (x, y)
        font_size：字體大小（預設 18pt，或可用 14pt）
        color：文字顏色（BGR 格式，預設黑色）
    
    回傳值：
        修改後的影像陣列（numpy array，BGR 色彩空間）
    
    工作原理：
        1. 若不支援中文或無字體，使用 OpenCV 內建英文字體
        2. 將 numpy BGR 陣列轉為 PIL RGB Image 物件
        3. 用 PIL 的中文字體在圖上繪製文字
        4. 將結果轉回 OpenCV 使用的 BGR numpy 陣列
    """
    
    # 檢查中文支援與字體是否可用
    if not CHINESE_SUPPORT or FONT is None:
        # 無中文支援時，使用 OpenCV 內建字體（英文）
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 1)  # 字體大小=0.6、顏色=color、線條厚度=1
        return frame
    
    try:
        # 步驟 1：將 numpy BGR 陣列轉換為 PIL RGB Image
        # cv2.cvtColor 執行色彩空間轉換（BGR→RGB）
        # Image.fromarray 將 numpy 陣列轉為 PIL Image 物件
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 步驟 2：建立繪圖物件（用於在 PIL Image 上繪製）
        draw = ImageDraw.Draw(pil_image)
        
        # 步驟 3：根據字體大小選擇合適的字體物件
        # 若 font_size==18 則用 FONT，否則用 FONT_SMALL
        font = FONT if font_size == 18 else FONT_SMALL
        
        # 步驟 4：在 PIL 圖像上繪製文字
        # PIL 使用 RGB 色彩空間，所以需要 reversed(color) 來轉換 BGR→RGB
        # fill 參數指定文字顏色
        draw.text(position, text, font=font, fill=tuple(reversed(color)))
        
        # 步驟 5：將 PIL Image 轉回 OpenCV 使用的 BGR numpy 陣列
        # np.array 將 PIL Image 轉為 numpy 陣列（RGB）
        # cv2.cvtColor 執行色彩空間轉換（RGB→BGR）
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 步驟 6：回傳修改後的影像陣列
        return frame
    
    except Exception as e:
        # 若中文渲染過程出錯，列印警告訊息
        print(f"[WARNING] Chinese rendering failed: {str(e)}")
        # 回退到 OpenCV 英文渲染作為備選方案
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 1)
        return frame


# ============================================================================
# 第四部分：環境驗證類別（工業級診斷）
# ============================================================================

class EnvironmentValidator:
    """
    環境驗證類別
    
    功能：
        ✓ 檢測虛擬環境（避免全域污染）
        ✓ 驗證必要套件（OpenCV、NumPy、PIL）
        ✓ 提供明確的錯誤提示與修復建議
    """
    
    @staticmethod
    def check_virtual_environment():
        """
        檢測程式是否在虛擬環境中執行
        
        工作原理：
            1. 讀取 VIRTUAL_ENV 環境變數
            2. 若存在表示在虛擬環境中
            3. 若不存在則提示使用者激活虛擬環境
        
        回傳值：
            True = 在虛擬環境中
            False = 不在虛擬環境中
        """
        # 從環境變數中取得虛擬環境路徑
        venv_path = os.getenv('VIRTUAL_ENV')
        
        # 若 VIRTUAL_ENV 存在，表示已激活虛擬環境
        if venv_path:
            print(f"[OK] Virtual environment detected: {venv_path}")
            return True
        else:
            # 虛擬環境未激活時的警告訊息
            print("[WARNING] Not in virtual environment")
            print("          Recommended activation: .venv\\Scripts\\activate")
            return False
    
    @staticmethod
    def check_dependencies():
        """
        驗證所有必要套件是否已安裝
        
        檢查清單：
            • cv2（OpenCV）：電腦視覺核心
            • numpy（NumPy）：數值計算基礎
            • PIL（Pillow）：中文文字渲染
        
        工作原理：
            1. 定義必要套件與其安裝名稱的對應關係
            2. 嘗試匯入每個套件
            3. 若匯入失敗則記錄為缺失
            4. 提供完整的安裝指令
        
        回傳值：
            True = 所有套件都已安裝
            False = 至少有一個套件缺失
        """
        # 列印驗證開始訊息
        print("\n[CHECK] Verifying required packages...\n")
        
        # 定義必要套件對應表
        # 鍵 = 匯入名稱、值 = pip 安裝名稱
        required_packages = {
            'cv2': 'opencv-python',      # OpenCV 庫
            'numpy': 'numpy',            # NumPy 陣列庫
            'PIL': 'pillow'              # PIL 圖像庫
        }
        
        # 初始化驗證結果旗標（預設全部通過）
        all_packages_ok = True
        
        # 遍歷所有必要套件並逐一檢查
        for import_name, package_name in required_packages.items():
            try:
                # 嘗試動態匯入套件
                __import__(import_name)
                # 若匯入成功則列印成功訊息
                print(f"  [OK] {package_name}")
            except ImportError:
                # 若匯入失敗則設定旗標並提供修復建議
                print(f"  [FAIL] {package_name} not installed")
                print(f"         Run: pip install {package_name}")
                all_packages_ok = False
        
        # 回傳整體驗證結果
        return all_packages_ok
    
    @staticmethod
    def validate():
        """
        執行完整的環境驗證流程
        
        驗證步驟：
            1. 檢測虛擬環境
            2. 驗證所有必要套件
            3. 若任何步驟失敗則終止程式
        
        回傳值：
            True = 環境驗證通過，可繼續執行
            False = 環境驗證失敗，應終止程式
        """
        # 列印驗證標題
        print("=" * 72)
        print("ENVIRONMENT VERIFICATION")
        print("=" * 72 + "\n")
        
        # 執行虛擬環境檢測
        venv_ok = EnvironmentValidator.check_virtual_environment()
        
        # 執行套件驗證
        deps_ok = EnvironmentValidator.check_dependencies()
        
        # 若套件驗證失敗，終止程式
        if not deps_ok:
            print("\n[FAIL] Missing required packages - Cannot proceed")
            return False
        
        # 驗證成功訊息
        print("\n[SUCCESS] Environment verification passed!\n")
        return True


# ============================================================================
# 第五部分：攝影機初始化與診斷類別
# ============================================================================

class CameraInitializer:
    """
    攝影機初始化與診斷類別
    
    功能：
        ✓ 掃描可用攝影機
        ✓ 智能開啟攝影機（支援 DirectShow 後退）
        ✓ 強制設定解析度（三次重試機制）
        ✓ 資源清理
    """
    
    def __init__(self):
        """
        初始化類別成員變數
        
        成員變數說明：
            cap：OpenCV VideoCapture 物件，代表攝影機
            actual_width：實際幀寬度（像素）
            actual_height：實際幀高度（像素）
            init_width：初始化時的幀寬度（駕動程式預設值）
            init_height：初始化時的幀高度（駕動程式預設值）
        """
        self.cap = None                 # 攝影機物件（未初始化）
        self.actual_width = None        # 實際寬度（未知）
        self.actual_height = None       # 實際高度（未知）
        self.init_width = None          # 初始寬度（未知）
        self.init_height = None         # 初始高度（未知）
    
    def scan_cameras(self):
        """
        掃描系統中所有可用的攝影機
        
        工作原理：
            1. 遍歷攝影機索引 0~9
            2. 嘗試開啟每台攝影機
            3. 記錄可成功開啟的攝影機
            4. 立即關閉攝影機（避免佔用）
        
        回傳值：
            available：可用攝影機索引的清單 [0, 1, ...]
        """
        # 列印掃描開始訊息
        print("[SCAN] Scanning available cameras...\n")
        
        # 初始化可用攝影機清單
        available = []
        
        # 嘗試索引 0 到 9（支援最多 10 台攝影機）
        for camera_index in range(10):
            try:
                # 嘗試開啟該索引的攝影機
                cap_test = cv2.VideoCapture(camera_index)
                
                # 檢查攝影機是否成功開啟
                if cap_test.isOpened():
                    # 成功開啟則加入清單
                    available.append(camera_index)
                    print(f"  [FOUND] Camera at index {camera_index}")
                    # 立即關閉以避免資源佔用
                    cap_test.release()
            except Exception as e:
                # 若發生例外則跳過該索引
                pass
        
        # 若未找到任何攝影機則列印警告
        if not available:
            print("  [WARNING] No cameras detected in system")
            return []
        
        # 列印掃描完成訊息
        print(f"\n[COMPLETE] Found {len(available)} camera(s) available\n")
        return available
    
    def open_camera(self, camera_index=0):
        """
        開啟指定索引的攝影機
        
        工作原理：
            1. 優先嘗試用 DirectShow 後端（CAP_DSHOW）
            2. 若失敗則退回預設 API
            3. 若仍失敗則提供除錯建議
        
        參數：
            camera_index：攝影機索引編號（預設=0）
        
        回傳值：
            True = 攝影機成功開啟
            False = 攝影機開啟失敗
        """
        # 列印連接嘗試訊息
        print(f"[CONNECT] Opening camera (index {camera_index})...\n")
        
        try:
            # 第一步：嘗試用 DirectShow 後端（最穩定）
            # CAP_DSHOW = DirectShow 後端（推薦用於 USB 攝影機）
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            
            # 檢查是否成功開啟
            if not self.cap.isOpened():
                # DirectShow 失敗時的提示訊息
                print("[RETRY] DirectShow backend failed, trying default API...\n")
                # 第二步：退回預設 API
                self.cap = cv2.VideoCapture(camera_index)
            
            # 再次檢查是否成功開啟
            if not self.cap.isOpened():
                # 開啟失敗的完整除錯訊息
                print("[FAIL] Cannot open camera\n")
                print("       Diagnostic suggestions:")
                print("       • Check USB cable connection")
                print("       • Check if camera is used by other program")
                print("       • Try different camera index (0 -> 1 or 2)")
                print("       • Update camera driver\n")
                return False
            
            # 成功開啟的確認訊息
            print("[SUCCESS] Camera opened successfully!\n")
            return True
        
        except Exception as e:
            # 捕捉任何發生的例外
            print(f"[ERROR] Exception occurred: {type(e).__name__}")
            print(f"        Details: {str(e)}\n")
            return False
    
    def detect_initial_resolution(self):
        """
        檢測攝影機的初始解析度（驅動程式預設值）
        
        工作原理：
            1. 從已開啟的攝影機讀取當前解析度
            2. 記錄初始寬高值
            3. 警告如果是低解析度（640x480）
        
        回傳值：
            True = 檢測成功
            False = 攝影機未開啟
        """
        # 檢查攝影機是否已開啟
        if self.cap is None or not self.cap.isOpened():
            return False
        
        # 讀取當前幀寬度（使用 CAP_PROP_FRAME_WIDTH 屬性）
        self.init_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 讀取當前幀高度（使用 CAP_PROP_FRAME_HEIGHT 屬性）
        self.init_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 列印檢測結果
        print(f"[DETECT] Initial resolution: {self.init_width} x {self.init_height}\n")
        
        # 檢查是否為低解析度（常見的驅動程式預設值）
        if self.init_width == 640 and self.init_height == 480:
            # 低解析度時的警告訊息
            print("[WARNING] 640x480 detected (may cause dark/blurry output)")
            print("          Attempting to upgrade to 1280x720\n")
        
        return True
    
    def force_resolution(self):
        """
        強制設定攝影機的輸出解析度
        
        工作原理：
            1. 三次重試機制（每次延遲 150ms 以穩定驅動程式）
            2. 每次重試後檢查實際解析度
            3. 若成功達到目標則提前返回
            4. 若失敗則記錄最終解析度
        
        回傳值：
            True = 成功設定為目標解析度
            False = 未能設定為目標解析度（但返回最接近的值）
        """
        # 檢查攝影機是否已開啟
        if self.cap is None or not self.cap.isOpened():
            return False
        
        # 列印解析度設定嘗試訊息
        print(f"[SET] Attempting to set resolution to {TARGET_WIDTH} x {TARGET_HEIGHT}...\n")
        
        # 三次重試迴圈
        for attempt_num in range(1, 4):
            # 設定目標寬度
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            # 設定目標高度
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            
            # 延遲 150ms 以允許驅動程式穩定
            time.sleep(0.15)
            
            # 讀取實際設定的寬度
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # 讀取實際設定的高度
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 列印本次嘗試的結果
            print(f"  Attempt {attempt_num}/3: {actual_width} x {actual_height}")
            
            # 檢查是否達到目標解析度
            if actual_width == TARGET_WIDTH and actual_height == TARGET_HEIGHT:
                # 成功達到目標則列印成功訊息並提前返回
                print(f"\n[SUCCESS] Resolution set successfully!\n")
                # 儲存實際寬度值
                self.actual_width = actual_width
                # 儲存實際高度值
                self.actual_height = actual_height
                return True
        
        # 若三次都失敗則列印警告訊息
        print(f"\n[WARNING] Failed to set target resolution")
        print(f"          Using: {actual_width} x {actual_height}\n")
        # 儲存實際寬度值（最接近的值）
        self.actual_width = actual_width
        # 儲存實際高度值（最接近的值）
        self.actual_height = actual_height
        return False
    
    def verify_and_initialize(self):
        """
        執行完整的攝影機初始化流程
        
        初始化步驟：
            1. 掃描可用攝影機
            2. 選擇可用攝影機並開啟
            3. 檢測初始解析度
            4. 強制設定目標解析度
        
        回傳值：
            True = 初始化成功
            False = 初始化失敗
        """
        # 步驟 1：掃描可用攝影機
        available_cameras = self.scan_cameras()
        
        # 若無可用攝影機則返回失敗
        if not available_cameras:
            return False
        
        # 選擇攝影機索引（優先使用配置值，若不可用則用第一台）
        selected_camera_index = CAM_INDEX if CAM_INDEX in available_cameras else available_cameras[0]
        
        # 步驟 2：開啟選定的攝影機
        if not self.open_camera(selected_camera_index):
            return False
        
        # 步驟 3：檢測初始解析度
        if not self.detect_initial_resolution():
            return False
        
        # 步驟 4：強制設定解析度
        self.force_resolution()
        
        # 初始化完成
        return True
    
    def cleanup(self):
        """
        釋放攝影機資源（工業級必須）
        
        工作原理：
            1. 檢查攝影機物件是否存在
            2. 檢查攝影機是否已開啟
            3. 呼叫 release() 釋放資源
        
        重要性：
            ✓ 防止攝影機被佔用（導致其他程式無法存取）
            ✓ 防止記憶體洩漏
            ✓ 防止系統無回應
        """
        # 檢查攝影機物件是否存在且已開啟
        if self.cap is not None and self.cap.isOpened():
            # 呼叫 release() 釋放攝影機資源
            self.cap.release()


# ============================================================================
# 第六部分：影像處理函式
# ============================================================================

def adjust_image(frame, brightness, contrast, saturation):
    """
    對影像進行三層調整：亮度、對比、飽和度
    
    參數說明：
        frame：輸入影像（numpy array，BGR 格式）
        brightness：亮度調整值（-100~+100）
        contrast：對比度倍數（0.5~3.0）
        saturation：飽和度倍數（0.0~2.0）
    
    工作原理：
        第一層（亮度）：直接加法調整像素值
        第二層（對比）：縮放像素值（控制明暗差異）
        第三層（飽和）：在 HSV 色彩空間調整色彩飽和度
    
    回傳值：
        result：調整後的影像（numpy array，BGR 格式）
    """
    
    # 第一步：亮度調整
    # convertScaleAbs 用於縮放並取絕對值
    # alpha=1.0（縮放因子）、beta=brightness（加性調整）
    adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)
    
    # 第二步：對比度調整
    # alpha=contrast（縮放所有像素值，使明暗差異擴大或縮小）
    # beta=0（不加性調整）
    adjusted = cv2.convertScaleAbs(adjusted, alpha=contrast, beta=0)
    
    # 第三步：飽和度調整
    # 轉換到 HSV 色彩空間（H=色相、S=飽和度、V=亮度）
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    
    # 分離 HSV 三個通道
    h, s, v = cv2.split(hsv)  # h=色相通道、s=飽和度通道、v=亮度通道
    
    # 調整飽和度通道（S）
    # 乘以 saturation 倍數來增加或減少飽和度
    s = cv2.convertScaleAbs(s, alpha=saturation, beta=0)
    
    # 重新合併三個通道
    hsv_adjusted = cv2.merge([h, s, v])
    
    # 轉換回 BGR 色彩空間（供 OpenCV 使用）
    result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    # 回傳調整後的影像
    return result


# ============================================================================
# 第七部分：Trackbar 回調函式
# ============================================================================

def on_trackbar_change(value):
    """
    Trackbar 事件回調函式
    
    功能說明：
        此函式在使用者拖動 Trackbar 時被呼叫
        實際的參數讀取在主迴圈中進行（使用 getTrackbarPos）
        此函式可保持空白（不需要立即處理）
    
    參數：
        value：Trackbar 的當前值（由 OpenCV 自動傳入）
    """
    # 此回調函式只是佔位符
    # 實際的參數處理在主迴圈中進行
    pass


# ============================================================================
# 第八部分：主程式函式
# ============================================================================

def main():
    """
    主程式入口點
    
    執行流程：
        1. 環境驗證
        2. 攝影機初始化
        3. 介面建設
        4. 主影像擷取迴圈
        5. 資源清理
    """
    
    # 列印程式標題
    print("\n" + "=" * 72)
    print("OpticalInspect - C270 Real-time Monitoring System v3.2")
    print("=" * 72 + "\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # 階段一：環境驗證
    # ════════════════════════════════════════════════════════════════════════
    
    # 執行環境驗證（檢測虛擬環境和套件）
    if not EnvironmentValidator.validate():
        # 環境驗證失敗則終止程式
        return
    
    # ════════════════════════════════════════════════════════════════════════
    # 階段二：攝影機初始化
    # ════════════════════════════════════════════════════════════════════════
    
    # 列印攝影機初始化階段標題
    print("=" * 72)
    print("CAMERA INITIALIZATION")
    print("=" * 72 + "\n")
    
    # 建立攝影機初始化器物件
    camera_initializer = CameraInitializer()
    
    # 執行完整初始化流程
    if not camera_initializer.verify_and_initialize():
        # 初始化失敗則列印錯誤訊息並終止
        print("[FAIL] Camera initialization failed - Cannot proceed\n")
        return
    
    # 取得已初始化的攝影機物件
    cap = camera_initializer.cap
    
    # ════════════════════════════════════════════════════════════════════════
    # 階段三：使用者介面初始化
    # ════════════════════════════════════════════════════════════════════════
    
    # 列印介面初始化階段標題
    print("=" * 72)
    print("UI INITIALIZATION")
    print("=" * 72 + "\n")
    
    # 列印視窗建設訊息
    print("[CREATE] Creating main window...")
    
    # 建立視窗
    # WINDOW_NORMAL 表示使用者可調整視窗大小
    cv2.namedWindow(WINDOW_NAME, WINDOW_MODE)
    
    # 調整視窗至預設大小（500x500）
    # 允許使用者自由拉伸
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH_INIT, WINDOW_HEIGHT_INIT)
    
    # 列印 Trackbar 建設訊息
    print("[CREATE] Creating adjustment trackbars...\n")
    
    # 建立亮度調整 Trackbar
    # 範圍：0-200（對應 -100~+100）
    # 初始值：BRIGHTNESS_INIT + 100（轉換為 0-200 範圍）
    cv2.createTrackbar("Brightness", WINDOW_NAME, 
                       BRIGHTNESS_INIT + 100, 200, on_trackbar_change)
    
    # 建立對比度調整 Trackbar
    # 範圍：0-300（對應 0.0~3.0）
    # 初始值：int(CONTRAST_INIT * 100)（轉換為 0-300 範圍）
    cv2.createTrackbar("Contrast", WINDOW_NAME, 
                       int(CONTRAST_INIT * 100), 300, on_trackbar_change)
    
    # 建立飽和度調整 Trackbar
    # 範圍：0-200（對應 0.0~2.0）
    # 初始值：int(SATURATION_INIT * 100)（轉換為 0-200 範圍）
    cv2.createTrackbar("Saturation", WINDOW_NAME, 
                       int(SATURATION_INIT * 100), 200, on_trackbar_change)
    
    # 列印介面初始化完成訊息
    print("[SUCCESS] UI initialization complete!\n")
    
    # ════════════════════════════════════════════════════════════════════════
    # 階段四：FPS 計算初始化
    # ════════════════════════════════════════════════════════════════════════
    
    # 紀錄 FPS 計算開始時間
    fps_calculation_start_time = time.time()
    # 在當前時間間隔內已讀取的幀數
    fps_frame_count = 0
    # 最後計算出的 FPS 值（初始=0）
    fps_display_value = 0.0
    
    # ════════════════════════════════════════════════════════════════════════
    # 階段五：主影像擷取迴圈
    # ════════════════════════════════════════════════════════════════════════
    
    # 列印主迴圈標題與使用說明
    print("=" * 72)
    print("REAL-TIME VIDEO STREAM")
    print("=" * 72)
    print("\n[START] Video capture loop started")
    print("Keys: ESC = quit | R = reset parameters")
    print("-" * 72 + "\n")
    
    try:
        # 主迴圈（無限迴圈，直到使用者按 ESC 或發生錯誤）
        while True:
            # ════════════════════────────────────────────────────────────
            # 步驟 1：讀取攝影機幀
            # ════════════════════────────────────────────────────────────
            
            # 從攝影機讀取一幀影像
            # ret = 讀取成功標誌（True/False）
            # frame = 讀取到的影像（numpy array，BGR）
            ret, frame = cap.read()
            
            # 檢查讀取是否成功
            if not ret:
                # 讀取失敗則列印警告並終止迴圈
                print("[WARNING] Failed to read frame from camera")
                break
            
            # ════════════════════────────────────────────────────────────
            # 步驟 2：讀取 Trackbar 調整值
            # ════════════════────────────────────────────────────────────
            
            # 讀取亮度 Trackbar 值（0-200），轉換為 -100~+100
            brightness_value = cv2.getTrackbarPos("Brightness", WINDOW_NAME) - 100
            
            # 讀取對比度 Trackbar 值（0-300），轉換為 0.0~3.0
            contrast_value = cv2.getTrackbarPos("Contrast", WINDOW_NAME) / 100.0
            
            # 讀取飽和度 Trackbar 值（0-200），轉換為 0.0~2.0
            saturation_value = cv2.getTrackbarPos("Saturation", WINDOW_NAME) / 100.0
            
            # ════════════════════────────────────────────────────────────
            # 步驟 3：應用影像調整
            # ════════────────────────────────────────────────────────────
            
            # 呼叫影像調整函式，應用所有調整
            display_frame = adjust_image(frame, brightness_value, 
                                        contrast_value, saturation_value)
            
            # ════════════════════────────────────────────────────────────
            # 步驟 4：FPS 計算
            # ════════════════════────────────────────────────────────────
            
            # 增加已讀幀數計數器
            fps_frame_count += 1
            
            # 計算自上次 FPS 計算以來已經過的時間
            fps_elapsed_time = time.time() - fps_calculation_start_time
            
            # 若已經過 1 秒則重新計算 FPS
            if fps_elapsed_time >= 1.0:
                # FPS = 過去 1 秒內的幀數 ÷ 已耗時間
                fps_display_value = fps_frame_count / fps_elapsed_time
                # 重置幀計數器
                fps_frame_count = 0
                # 重置計時起點（重新開始 1 秒計時）
                fps_calculation_start_time = time.time()
            
            # ════════════════════────────────────────────────────────────
            # 步驟 5：獲取當前日期時間
            # ════════────────────────────────────────────────────────────
            
            # 取得當前時刻
            current_time = datetime.now()
            
            # 格式化為「YYYY/MM/DD」格式
            date_display_string = current_time.strftime("%Y/%m/%d")
            
            # 格式化為「HH:MM:SS」格式
            time_display_string = current_time.strftime("%H:%M:%S")
            
            # ════════════════════────────────────────────────────────────
            # 步驟 6：在影像上繪製判讀資訊（黑色字體，整齊排列）
            # ════════════════════────────────────────────────────────────
            
            # 計數器：用於追蹤當前繪製的行號
            current_line_number = 0
            
            # —— 第 1 行：FPS —— 
            # 計算本行的 Y 座標
            current_y_position = INFO_START_Y + (current_line_number * INFO_LINE_SPACING)
            # 繪製 FPS 文字（黑色）
            cv2.putText(display_frame, f"FPS: {fps_display_value:.1f}",
                       (INFO_START_X, current_y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLACK, 2)
            # 行號遞增
            current_line_number += 1
            
            # —— 第 2 行：日期 ——
            current_y_position = INFO_START_Y + (current_line_number * INFO_LINE_SPACING)
            cv2.putText(display_frame, f"Date: {date_display_string}",
                       (INFO_START_X, current_y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1)
            current_line_number += 1
            
            # —— 第 3 行：時間 ——
            current_y_position = INFO_START_Y + (current_line_number * INFO_LINE_SPACING)
            cv2.putText(display_frame, f"Time: {time_display_string}",
                       (INFO_START_X, current_y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1)
            current_line_number += 1
            
            # —— 第 4 行：亮度 ——
            current_y_position = INFO_START_Y + (current_line_number * INFO_LINE_SPACING)
            cv2.putText(display_frame, f"Brightness: {brightness_value:+.0f}",
                       (INFO_START_X, current_y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1)
            current_line_number += 1
            
            # —— 第 5 行：對比度 ——
            current_y_position = INFO_START_Y + (current_line_number * INFO_LINE_SPACING)
            cv2.putText(display_frame, f"Contrast: {contrast_value:.2f}x",
                       (INFO_START_X, current_y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1)
            current_line_number += 1
            
            # —— 第 6 行：飽和度 ——
            current_y_position = INFO_START_Y + (current_line_number * INFO_LINE_SPACING)
            cv2.putText(display_frame, f"Saturation: {saturation_value:.2f}x",
                       (INFO_START_X, current_y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1)
            current_line_number += 1
            
            # —— 第 7 行：解析度 ——
            # 獲取目前幀的高度與寬度
            frame_height = display_frame.shape[0]  # 行數 = 高度
            frame_width = display_frame.shape[1]   # 列數 = 寬度
            
            current_y_position = INFO_START_Y + (current_line_number * INFO_LINE_SPACING)
            cv2.putText(display_frame, f"Size: {frame_width} x {frame_height}",
                       (INFO_START_X, current_y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1)
            
            # ════════════════════────────────────────────────────────────
            # 步驟 7：在視窗中顯示調整後的影像
            # ════════────────────────────────────────────────────────────
            
            # 呼叫 imshow 在視窗中顯示影像
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # ════════════════════────────────────────────────────────────
            # 步驟 8：鍵盤事件處理
            # ════════════════════────────────────────────────────────────
            
            # 等待 1 毫秒的鍵盤輸入
            # waitKey 回傳所按按鍵的 ASCII 碼（若無按鍵則傳 -1）
            # & 0xFF：只取最低 8 位元（處理平台相容性）
            key_pressed = cv2.waitKey(1) & 0xFF
            
            # —— 檢查 ESC 鍵（ASCII 碼 27）——
            if key_pressed == 27:
                # ESC 鍵被按下則列印訊息並跳出迴圈
                print("\n[QUIT] ESC key detected - Exiting...")
                break
            
            # —— 檢查 R 鍵（用於重置所有參數）——
            elif key_pressed == ord('r'):
                # 重置亮度 Trackbar 至初始值
                cv2.setTrackbarPos("Brightness", WINDOW_NAME, BRIGHTNESS_INIT + 100)
                # 重置對比度 Trackbar 至初始值
                cv2.setTrackbarPos("Contrast", WINDOW_NAME, int(CONTRAST_INIT * 100))
                # 重置飽和度 Trackbar 至初始值
                cv2.setTrackbarPos("Saturation", WINDOW_NAME, int(SATURATION_INIT * 100))
                # 列印重置確認訊息
                print("[RESET] All parameters reset to defaults")
    
    # ════════════════════────────────────────────────────────────────────
    # 異常處理：Ctrl+C 中斷
    # ════════════════════────────────────────────────────────────────────
    
    except KeyboardInterrupt:
        # 若使用者按 Ctrl+C 則列印中斷訊息
        print("\n[INTERRUPT] Ctrl+C detected - Shutting down...")
    
    # ════════════════════────────────────────────────────────────────────
    # 異常處理：捕捉其他例外
    # ════════════════════────────────────────────────────────────────────
    
    except Exception as e:
        # 捕捉任何未預期的例外
        print(f"\n[ERROR] Unexpected exception: {type(e).__name__}")
        print(f"        Details: {str(e)}")
    
    # ════════════════════────────────────────────────────────────────────
    # 最終階段：資源清理（工業級必須）
    # ════════════════════────────────────────────────────────────────────
    
    finally:
        # 無論程式如何終止，此區塊必定執行（保證資源釋放）
        
        # 列印清理階段標題
        print("\n" + "=" * 72)
        print("RESOURCE CLEANUP")
        print("=" * 72 + "\n")
        
        # 列印開始清理訊息
        print("[CLEANUP] Releasing resources...\n")
        
        # —— 清理 1：釋放攝影機資源 ——
        if cap is not None and cap.isOpened():
            # 呼叫 release() 釋放攝影機佔用的資源
            cap.release()
            # 列印完成訊息
            print("[DONE] Camera resource released")
        
        # —— 清理 2：關閉所有視窗 ——
        # destroyAllWindows 會關閉由 OpenCV 建立的所有視窗
        cv2.destroyAllWindows()
        # 列印完成訊息
        print("[DONE] All display windows closed")
        
        # —— 完成訊息 ——
        print("\n[SUCCESS] Program terminated safely")
        print("=" * 72 + "\n")


# ============================================================================
# 第九部分：程式進入點
# ============================================================================

if __name__ == "__main__":
    """
    程式進入點判斷
    
    __name__ 是 Python 的特殊變數：
        • 若此檔案直接執行：__name__ = "__main__"
        • 若此檔案被其他模組 import：__name__ = "模組名"
    
    此判斷確保 main() 只在直接執行時呼叫，不會在被 import 時執行
    """
    # 呼叫主程式函式
    main()
