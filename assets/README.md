# App Assets

This directory contains assets for the macOS application.

## App Icon

To add a custom app icon:

1. **Create an icon file:**
   - Create a 1024x1024 PNG image
   - Use an icon design tool or online service

2. **Convert to ICNS format:**
   ```bash
   # Create iconset directory
   mkdir icon.iconset

   # Generate different sizes (macOS requires multiple sizes)
   sips -z 16 16     icon_1024.png --out icon.iconset/icon_16x16.png
   sips -z 32 32     icon_1024.png --out icon.iconset/icon_16x16@2x.png
   sips -z 32 32     icon_1024.png --out icon.iconset/icon_32x32.png
   sips -z 64 64     icon_1024.png --out icon.iconset/icon_32x32@2x.png
   sips -z 128 128   icon_1024.png --out icon.iconset/icon_128x128.png
   sips -z 256 256   icon_1024.png --out icon.iconset/icon_128x128@2x.png
   sips -z 256 256   icon_1024.png --out icon.iconset/icon_256x256.png
   sips -z 512 512   icon_1024.png --out icon.iconset/icon_256x256@2x.png
   sips -z 512 512   icon_1024.png --out icon.iconset/icon_512x512.png
   sips -z 1024 1024 icon_1024.png --out icon.iconset/icon_512x512@2x.png

   # Convert to ICNS
   iconutil -c icns icon.iconset -o icon.icns

   # Clean up
   rm -rf icon.iconset
   ```

3. **Place the icon:**
   - Save the `icon.icns` file in this `assets/` directory
   - The setup.py script will automatically include it in the app bundle

## Using Online Icon Generators

If you don't want to use command line:

1. Use an online service like:
   - https://cloudconvert.com/png-to-icns
   - https://iconverticons.com/online/

2. Upload your 1024x1024 PNG
3. Download the generated .icns file
4. Place it in this directory as `icon.icns`

## Icon Design Tips

- Use a simple, recognizable design
- Avoid fine details (they don't scale well)
- Use high contrast colors
- Center your design
- Test at different sizes
- Consider macOS design guidelines

## Default Icon

If no icon is provided, macOS will use the default Python app icon.
