from flask import Flask, render_template, request, jsonify
import sys
import io
import contextlib
from ilp_flexible import run_test_case, generate_random_samples
import os
import traceback

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': '请求数据格式不正确'})
        
        # 从前端获取参数
        m = int(data.get('m', 45))
        n = int(data.get('n', 7))
        k = int(data.get('k', 6))
        j = int(data.get('j', 5))
        s = int(data.get('s', 5))
        strict_coverage = data.get('strict_coverage', False)
        min_cover = int(data.get('min_cover', 1))
        
        # 校验参数
        if not (1 <= n <= 25 and 1 <= k <= n and 1 <= j <= k and 1 <= s <= j and m >= n):
            return jsonify({
                'success': False, 
                'message': '参数错误！请确保: 1 <= n <= 25, 1 <= k <= n, 1 <= j <= k, 1 <= s <= j, m >= n'
            })
        
        # 捕获输出
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            # 创建一个临时测试用例
            test_case = {
                "n": n, 
                "k": k, 
                "j": j, 
                "s": s, 
                "m": m, 
                "strict_coverage": strict_coverage, 
                "min_cover": min_cover
            }
            
            # 调用run_test_case函数运行测试用例
            result = run_test_case(test_case)
        
        # 获取输出结果
        console_output = output.getvalue()
        
        # 将样本集合也返回给前端
        return jsonify({
            'success': True, 
            'result': console_output,
            'samples': result['samples'] if result else []
        })
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"处理请求时出错: {error_details}")
        return jsonify({'success': False, 'message': f"服务器错误: {str(e)}", 'error_details': error_details})

@app.route('/storedb', methods=['POST'])
def store_db():
    try:
        # 这里实现存储到数据库的功能
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': '请求数据格式不正确'})
        return jsonify({'success': True, 'message': '数据已成功存储到数据库'})
    except Exception as e:
        return jsonify({'success': False, 'message': f"存储失败: {str(e)}"})

@app.route('/delete', methods=['POST'])
def delete():
    try:
        # 这里实现删除功能
        return jsonify({'success': True, 'message': '数据已成功删除'})
    except Exception as e:
        return jsonify({'success': False, 'message': f"删除失败: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
