#pragma once



BYTE* LoadBMP(int* width, int* height, long* size, LPCTSTR bmpfile);
BYTE* ConvertBMPToRGBBuffer(BYTE* Buffer, int width, int height);