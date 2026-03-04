import subprocess
import asyncio
from pyppeteer import launcher
async def main():
    browser = await launcher.launch()
    page = await browser._newPage()
     # code ...
     
