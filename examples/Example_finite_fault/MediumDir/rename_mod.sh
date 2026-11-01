#!/bin/bash

# 检查是否有文件参数
if [ $# -eq 0 ]; then
    echo "请提供要处理的文件或使用通配符"
    echo "用法: $0 文件1 文件2 ..."
    echo "示例: $0 *.txt"
    exit 1
fi

# 处理每个文件
for file in "$@"; do
    # 检查文件是否存在
    if [ ! -e "$file" ]; then
        echo "跳过不存在的文件: $file"
        continue
    fi

    # 获取文件名和扩展名
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    name="${filename%.*}"

    # 如果文件名中没有扩展名，则name会等于filename
    if [ "$name" = "$filename" ]; then
        extension=""
    else
        extension=".${extension}"
    fi

    # 检查文件名是否符合"经度_纬度"格式
    if [[ $name =~ ^([0-9.]+)_([0-9.]+)$ ]]; then
        longitude="${BASH_REMATCH[1]}"
        latitude="${BASH_REMATCH[2]}"
        
        # 添加小数点和一位小数
        new_longitude=$(printf "%.1f" "$longitude")
        new_latitude=$(printf "%.1f" "$latitude")
        
        # 构建新文件名
        new_name="${new_longitude}_${new_latitude}${extension}"
        new_file="${new_name}"
        
        # 重命名文件
        mv -v "$file" "$new_file"
    else
        echo "跳过不符合格式的文件: $file"
    fi
done

echo "处理完成"    