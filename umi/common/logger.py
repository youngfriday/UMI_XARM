"""
Simple colorful logging utility for UMI project
Usage:
    from umi.common.logger import Logger
    
    Logger.info("Robot initializing...")
    Logger.success("Movement successful")
    Logger.warning("Queue might be full")
    Logger.error("Connection failed")
"""

import time

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

class Logger:
    """Simple colorful logger"""
    
    enabled = True
    show_timestamp = True
    
    @classmethod
    def _log(cls, msg: str, color: str = "", prefix: str = ""):
        """Internal logging method"""
        if not cls.enabled:
            return
            
        timestamp = f"{time.time():.3f} " if cls.show_timestamp else ""
        
        if COLORAMA_AVAILABLE:
            if timestamp:
                timestamp = f"{Fore.CYAN}{timestamp}{Style.RESET_ALL}"
            message = f"{timestamp}{color}{prefix} {msg}{Style.RESET_ALL}"
        else:
            message = f"{timestamp}{prefix} {msg}"
        
        print(message)
    
    @classmethod
    def info(cls, msg: str):
        """Info level logging (blue)"""
        if COLORAMA_AVAILABLE:
            cls._log(msg, Fore.BLUE, "ℹ️ ")
        else:
            cls._log(msg, "", "[INFO]")
    
    @classmethod
    def success(cls, msg: str):
        """Success level logging (green)"""
        if COLORAMA_AVAILABLE:
            cls._log(msg, Fore.GREEN, "✓")
        else:
            cls._log(msg, "", "[SUCCESS]")
    
    @classmethod
    def warning(cls, msg: str):
        """Warning level logging (yellow)"""
        if COLORAMA_AVAILABLE:
            cls._log(msg, Fore.YELLOW, "⚠️ ")
        else:
            cls._log(msg, "", "[WARNING]")
    
    @classmethod
    def error(cls, msg: str):
        """Error level logging (red)"""
        if COLORAMA_AVAILABLE:
            cls._log(msg, Fore.RED, "❌")
        else:
            cls._log(msg, "", "[ERROR]")
    
    @classmethod
    def debug(cls, msg: str):
        """Debug level logging (white)"""
        if COLORAMA_AVAILABLE:
            cls._log(msg, Fore.WHITE, "🐛")
        else:
            cls._log(msg, "", "[DEBUG]")