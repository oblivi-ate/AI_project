// 获取页面元素
const storeDBBtn = document.getElementById('storeDB');
const executeBtn = document.getElementById('execute');
const deleteBtn = document.getElementById('deleteBtn');
const printBtn = document.getElementById('print');
const nextBtn = document.getElementById('next');
const valueInput = document.getElementById('valueInput');
const results = document.getElementById('results');

// 为每个用户输入框绑定事件
document.querySelectorAll('.user-input').forEach(input => {
  input.addEventListener('input', function () {
    console.log(`Input ${this.id} changed to: ${this.value}`);
    // 在这里添加您希望在输入框值变化时执行的操作
  });
});

// 这里可以添加各个按钮点击事件的处理逻辑
storeDBBtn.addEventListener('click', function () {
  // 存储到数据库的逻辑，这里先简单提示
  alert('Store to DB function clicked');
});

executeBtn.addEventListener('click', function () {
  // 执行的逻辑，这里先简单提示
  alert('Execute function clicked');
});

deleteBtn.addEventListener('click', function () {
  // 删除的逻辑，这里先简单提示
  alert('Delete function clicked');
});

printBtn.addEventListener('click', function () {
  // 打印的逻辑，这里先简单提示
  alert('Print function clicked');
});

nextBtn.addEventListener('click', function () {
  // 下一步的逻辑，这里先简单提示
  alert('Next function clicked');
});

document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const executeBtn = document.getElementById('execute');
    const storeDBBtn = document.getElementById('storeDB');
    const deleteBtn = document.getElementById('deleteBtn');
    const printBtn = document.getElementById('print');
    const nextBtn = document.getElementById('next');
    const randomNOption = document.getElementById('randomN');
    const inputNOption = document.getElementById('inputN');
    const userInputs = document.querySelectorAll('.user-input');
    const resultsTextarea = document.getElementById('results');
    const valueInputTextarea = document.getElementById('valueInput');
    
    // 默认选择随机N选项
    randomNOption.checked = true;
    
    // 处理随机N和输入N选项的切换
    randomNOption.addEventListener('change', function() {
        if (this.checked) {
            disableUserInputs();
        }
    });
    
    inputNOption.addEventListener('change', function() {
        if (this.checked) {
            enableUserInputs();
        }
    });
    
    // 执行按钮点击事件
    executeBtn.addEventListener('click', function(e) {
        e.preventDefault();
        
        // 获取参数
        const m = parseInt(document.getElementById('m').value) || 45;
        const n = parseInt(document.getElementById('n').value) || 7;
        const k = parseInt(document.getElementById('k').value) || 6;
        const j = parseInt(document.getElementById('j').value) || 5;
        const s = parseInt(document.getElementById('s').value) || 4;
        
        // 验证参数
        if (!validateParameters(m, n, k, j, s)) {
            return;
        }
        
        // 获取N选项
        const nOption = randomNOption.checked ? 'randomN' : 'inputN';
        
        // 如果是输入N选项，收集用户输入的值
        const userValues = [];
        if (nOption === 'inputN') {
            for (let i = 1; i <= n; i++) {
                const input = document.getElementById(`userInput${i}`);
                if (input && input.value.trim()) {
                    userValues.push(input.value.trim());
                } else {
                    alert(`请为所有${n}个样本输入值`);
                    return;
                }
            }
        }
        
        // 构造请求数据
        const requestData = {
            m: m,
            n: n,
            k: k,
            j: j,
            s: s,
            nOption: nOption,
            userValues: userValues,
            strict_coverage: false,
            min_cover: 1
        };
        
        // 显示加载状态
        resultsTextarea.value = "计算中...";
        
        // 发送请求到后端
        fetch('/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 显示结果
                resultsTextarea.value = data.result;
                
                // 如果有样本，显示在值输入框中
                if (data.samples && data.samples.length > 0) {
                    console.log("收到样本集:", data.samples);
                    valueInputTextarea.value = data.samples.join(', ');
                    
                    // 如果用户选择了输入N选项，则还需要填充用户输入框
                    if (inputNOption.checked) {
                        const samples = data.samples;
                        for (let i = 0; i < Math.min(samples.length, n); i++) {
                            const input = document.getElementById(`userInput${i+1}`);
                            if (input) {
                                input.value = samples[i];
                            }
                        }
                    }
                } else {
                    console.warn("未收到样本集数据");
                    valueInputTextarea.value = "未收到样本集数据";
                }
            } else {
                resultsTextarea.value = data.message || "执行出错";
                if (data.error_details) {
                    console.error("错误详情:", data.error_details);
                }
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultsTextarea.value = "请求错误: " + error;
        });
    });
    
    // 存储到数据库按钮点击事件
    storeDBBtn.addEventListener('click', function(e) {
        e.preventDefault();
        
        // 获取当前表单数据和结果
        const m = parseInt(document.getElementById('m').value) || 45;
        const n = parseInt(document.getElementById('n').value) || 7;
        const k = parseInt(document.getElementById('k').value) || 6;
        const j = parseInt(document.getElementById('j').value) || 5;
        const s = parseInt(document.getElementById('s').value) || 4;
        
        const result = resultsTextarea.value;
        const valueInput = valueInputTextarea.value;
        
        // 构造请求数据
        const requestData = {
            m: m,
            n: n,
            k: k,
            j: j,
            s: s,
            result: result,
            valueInput: valueInput
        };
        
        // 发送请求到后端
        fetch('/storedb', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message || "操作成功");
        })
        .catch(error => {
            console.error('Error:', error);
            alert("存储失败: " + error);
        });
    });
    
    // 删除按钮点击事件
    deleteBtn.addEventListener('click', function(e) {
        e.preventDefault();
        
        if (confirm("确定要删除当前数据吗？")) {
            fetch('/delete', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                alert(data.message || "删除成功");
                // 清空输入框和结果
                clearForm();
            })
            .catch(error => {
                console.error('Error:', error);
                alert("删除失败: " + error);
            });
        }
    });
    
    // 打印按钮点击事件
    printBtn.addEventListener('click', function(e) {
        e.preventDefault();
        window.print();
    });
    
    // 下一步按钮点击事件
    nextBtn.addEventListener('click', function(e) {
        e.preventDefault();
        // 这里可以添加下一步的逻辑
        alert("进入下一步");
    });
    
    // 辅助函数
    function validateParameters(m, n, k, j, s) {
        if (isNaN(m) || isNaN(n) || isNaN(k) || isNaN(j) || isNaN(s)) {
            alert("请输入有效的数字参数");
            return false;
        }
        
        if (!(1 <= n && n <= 25)) {
            alert("n 必须在 1 到 25 之间");
            return false;
        }
        
        if (!(1 <= k && k <= n)) {
            alert("k 必须在 1 到 n 之间");
            return false;
        }
        
        if (!(1 <= j && j <= k)) {
            alert("j 必须在 1 到 k 之间");
            return false;
        }
        
        if (!(1 <= s && s <= j)) {
            alert("s 必须在 1 到 j 之间");
            return false;
        }
        
        if (m < n) {
            alert("m 必须大于或等于 n");
            return false;
        }
        
        return true;
    }
    
    function disableUserInputs() {
        userInputs.forEach(input => {
            input.disabled = true;
            input.classList.add('opacity-50');
        });
    }
    
    function enableUserInputs() {
        const n = parseInt(document.getElementById('n').value) || 7;
        
        userInputs.forEach((input, index) => {
            if (index < n) {
                input.disabled = false;
                input.classList.remove('opacity-50');
            } else {
                input.disabled = true;
                input.classList.add('opacity-50');
            }
        });
    }
    
    function clearForm() {
        // 清空所有输入框
        document.getElementById('m').value = '';
        document.getElementById('n').value = '';
        document.getElementById('k').value = '';
        document.getElementById('j').value = '';
        document.getElementById('s').value = '';
        
        // 清空文本区域
        resultsTextarea.value = '';
        valueInputTextarea.value = '';
        
        // 清空用户输入
        userInputs.forEach(input => {
            input.value = '';
        });
    }
    
    // 初始化时禁用用户输入框
    disableUserInputs();
});