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