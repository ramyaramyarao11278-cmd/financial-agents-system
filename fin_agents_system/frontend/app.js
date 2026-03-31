document.addEventListener('DOMContentLoaded', function() {
        // 輔助函數：把任何日期字串轉成純 YYYY-MM-DD
    function normalizeDate(dateStr) {
        if (!dateStr) return '';
        // 處理常見格式：2026-02-12、2026-02-12 00:00:00、2026-02-12T00:00:00、2026/02/12 等
        const str = String(dateStr).trim();
        const cleaned = str.split(' ')[0].split('T')[0].replace(/\//g, '-');
        // 驗證格式 yyyy-mm-dd
        if (/^\d{4}-\d{2}-\d{2}$/.test(cleaned)) {
            return cleaned;
        }
        return ''; // 無效就返回空字串，後面會過濾掉
    }
    // 1. 更新时间戳
    function updateTimestamp() {
        const updateTime = document.getElementById('updateTime');
        updateTime.textContent = new Date().toLocaleString('zh-CN');
    }
    
    // 设置初始时间戳并每分钟更新
    updateTimestamp();
    setInterval(updateTimestamp, 60000);
    
    // 2. WebSocket连接管理
    let ws;
    function initWebSocket() {
        // 关闭现有连接（如果存在）
        if (ws) {
            ws.close();
        }
        
        // 创建新的WebSocket连接
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.hostname}:8000/ws`;
        ws = new WebSocket(wsUrl);
        
        ws.onopen = function() {
            console.log('WebSocket连接已建立');
        };
        
        ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'agent_update') {
                    handleAgentUpdate(data);
                } else if (data.type === 'analysis_complete') {
                    handleAnalysisComplete(data);
                } else if (data.type === 'connection_status') {
                    console.log('连接状态:', data);
                }
            } catch (error) {
                console.error('解析WebSocket消息失败:', error);
            }
        };
        
        ws.onclose = function() {
            console.log('WebSocket连接已关闭，尝试重新连接...');
            // 尝试重新连接
            setTimeout(initWebSocket, 3000);
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket错误:', error);
        };
    }
    
    // 处理智能体状态更新
    function handleAgentUpdate(data) {
        const { agent_name, status, message } = data;
        console.log('智能体更新:', data);
        
        // 更新加载文本
        const loadingText = document.getElementById('loadingText');
        if (loadingText) {
            loadingText.textContent = `${agent_name}: ${message}`;
        }
        
        // 更新工作流步骤状态
        updateWorkflowStep(agent_name, status);
    }
    
    // 更新工作流步骤状态
    function updateWorkflowStep(agent_name, status) {
        // 映射智能体名称到步骤ID
        const agentStepMap = {
            '数据工程师': 'step1',
            '情感分析师': 'step2',
            '技术分析师': 'step3',
            '回测专家': 'step4',
            '回测验证': 'step4',
            '策略生成器': 'step5',
            '结果存储': 'step5'
        };
        
        const stepId = agentStepMap[agent_name];
        if (!stepId) return;
        
        const step = document.getElementById(stepId);
        if (!step) return;
        
        // 更新步骤状态
        if (status === 'running') {
            // 移除其他步骤的active类
            const allSteps = document.querySelectorAll('.step');
            allSteps.forEach(s => s.classList.remove('active'));
            // 添加当前步骤的active类
            step.classList.remove('completed');
            step.classList.add('active');
        } else if (status === 'completed') {
            step.classList.remove('active');
            step.classList.add('completed');
            // 激活下一个步骤（如果存在）
            const nextStep = step.nextElementSibling;
            if (nextStep && nextStep.classList.contains('step')) {
                nextStep.classList.add('active');
            }
        }
    }
    
    // 初始化WebSocket连接
    initWebSocket();
    
    // 2. 初始化图表
    // 情感分析饼图 - 默认值修改为测试用
    const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
    const sentimentChart = new Chart(sentimentCtx, {
        type: 'pie',
        data: {
            labels: ['正面', '中性', '负面'],
            datasets: [{
                data: [30, 40, 30], // 默认全部为0，测试时会被后端数据替换
                backgroundColor: [
                    '#10b981', // 绿色 - 正面
                    '#f59e0b', // 橙色 - 中性
                    '#ef4444'  // 红色 - 负面
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    align: 'center',
                    labels: {
                        padding: 20,
                        font: {
                            size: 14
                        },
                        usePointStyle: true,
                        boxWidth: 10
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            return `${label}: ${value}条`;
                        }
                    }
                }
            }
        }
    });
    
    // 投资组合净值走势图 - 默认值修改为测试用
    const portfolioCtx = document.getElementById('portfolioChart').getContext('2d');
    const portfolioChart = new Chart(portfolioCtx, {
        type: 'line',
        data: {
            labels: ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05', '2026-01-06', '2026-01-07'],
            datasets: [{
                label: '投资组合净值',
                data: [100000, 100000, 100000, 100000, 100000, 100000, 100000], // 默认平线，测试时会被后端数据替换
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.2,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const value = context.raw || 0;
                            return `净值: $${value.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 45
                    }
                },
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
    
    // 3. 生成交易信号时间线
    function generateSignals(signalsData) {
    const signalsList = document.getElementById('signals-list');
    signalsList.innerHTML = '';

    const sortedSignals = signalsData
        .map(s => ({
            ...s,
            date: normalizeDate(s.date),           // 清理日期
            ts: s.date ? new Date(normalizeDate(s.date)).getTime() : NaN
        }))
        .filter(s => s.date && !isNaN(s.ts))     // 移除無效日期
        .sort((a, b) => a.ts - b.ts);            // 嚴格按時間升序

    if (sortedSignals.length === 0) {
        signalsList.innerHTML = '<div style="text-align:center;padding:20px;color:#9ca3af;">無有效交易信號</div>';
        return;
    }

    sortedSignals.forEach(signal => {
        const signalItem = document.createElement('div');
        signalItem.className = 'signal-item';

        const signalClass = signal.signal === 'buy' ? 'signal-buy' : 'signal-sell';
        const signalText = signal.signal === 'buy' ? '买入' : '卖出';

        signalItem.innerHTML = `
            <div class="signal-date">${signal.date}</div>
            <div class="signal-type ${signalClass}">${signalText}</div>
            <div class="tooltip">
                <strong>${signal.indicator}指标:</strong> ${signal.value.toFixed(2)}
                <span class="tooltiptext">${signal.indicator}模型生成${signalText}信号，指标值为${signal.value.toFixed(2)}</span>
            </div>
        `;

        signalsList.appendChild(signalItem);
    });
}
    
    // 初始信号数据
    const initialSignals = [
        { date: "2026-01-10", signal: "buy", indicator: "ppo", value: 112.5 },
        { date: "2026-01-11", signal: "buy", indicator: "ppo", value: 113 },
        { date: "2026-01-12", signal: "buy", indicator: "ppo", value: 113.5 },
        { date: "2026-01-13", signal: "buy", indicator: "ppo", value: 114 },
        { date: "2026-01-14", signal: "buy", indicator: "ppo", value: 114.5 },
        { date: "2026-01-15", signal: "buy", indicator: "ppo", value: 115 }
    ];
    
    generateSignals(initialSignals);
    
    // 4. 历史记录管理
    function updateHistory(analysisData) {
        const historyContent = document.getElementById('history-content');
        
        const historyItem = document.createElement('div');
        historyItem.style.cssText = `
            width: 100%;
            background-color: #f0f9ff;
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        `;
        
        historyItem.innerHTML = `
            <div style="font-weight: 600; color: #1e3a8a; margin-bottom: 8px; font-size: 16px;">${new Date().toLocaleString('zh-CN')} - 情感分析+技术回测完成</div>
            <div style="font-size: 16px; color: #4b5563;">
                <span style="margin-right: 16px;"><i class="fas fa-smile" style="color: #10b981;"></i> 情感得分: ${analysisData.sentiment_score || 0.64} (${analysisData.sentiment_classification || '积极'})</span>
                <span style="margin-right: 16px;"><i class="fas fa-chart-line" style="color: #3b82f6;"></i> 回测收益: +${analysisData.total_return || 2.12}%</span>
                <span><i class="fas fa-bell" style="color: #f59e0b;"></i> 交易信号: ${analysisData.signal_count || 6}个买入</span>
            </div>
        `;
        
        // 如果是第一条记录，替换默认文本
        if (historyContent.innerHTML.trim() === '<p>暂无历史记录</p>') {
            historyContent.innerHTML = '';
        }
        
        historyContent.appendChild(historyItem);
    }
    
    // 初始历史记录
    updateHistory({});
    
    // 5. 时间范围选择交互
    const timeOptions = document.querySelectorAll('.time-option');

    timeOptions.forEach(option => {
        option.addEventListener('click', function() {

            // 只影响当前按钮组
            const parent = this.closest('.time-options');

            parent.querySelectorAll('.time-option')
            .forEach(opt => opt.classList.remove('active'));

            this.classList.add('active');

            // ⭐ 关键：重新刷新图表
            if (window.lastAnalysisData) {
                handleAnalysisComplete(window.lastAnalysisData);
            }

        });
    });
    
    // 处理分析完成的WebSocket消息
    function handleAnalysisComplete(data) {

        window.lastAnalysisData = data;
        // 隐藏分析状态提示
        const analysisStatus = document.getElementById('analysisStatus');
        
        if (data.status === 'completed') {
            // 显示成功提示
            showNotification('分析完成! 已更新最新结果。', 'success');
            
            // 更新图表数据
            if (data.results) {
                // 修复：获取正确的结果层级
                const actualResults = data.results.results || data.results;
                
                // 处理情感分析数据
                const sentiment = actualResults.sentiment || {};
                
                // 修复：根据后端实际返回的数据结构获取情感分布
                let positiveCount = 0;
                let neutralCount = 0;
                let negativeCount = 0;
                
                // 情况1：如果有sentiment_breakdown字段（旧格式）
                if (sentiment.sentiment_breakdown && Array.isArray(sentiment.sentiment_breakdown)) {
                    positiveCount = sentiment.sentiment_breakdown.filter(item => item.sentiment_label === 'positive').length;
                    neutralCount = sentiment.sentiment_breakdown.filter(item => item.sentiment_label === 'neutral').length;
                    negativeCount = sentiment.sentiment_breakdown.filter(item => item.sentiment_label === 'negative').length;
                }
                // 情况2：如果有sentiment_dimensions字段（新格式）
                else if (sentiment.sentiment_dimensions && typeof sentiment.sentiment_dimensions === 'object') {
                    // 从sentiment_dimensions中计算情感分布
                    const dimensions = Object.values(sentiment.sentiment_dimensions);
                    if (dimensions.length > 0) {
                        // 使用平均tone值作为情感判断依据
                        const avgTone = dimensions.reduce((sum, dim) => sum + (dim.avg_tone || 0), 0) / dimensions.length;
                        
                        // 根据平均tone值设置情感分布
                        if (avgTone > 0.3) {
                            positiveCount = dimensions.length;
                            neutralCount = 0;
                            negativeCount = 0;
                        } else if (avgTone < -0.3) {
                            positiveCount = 0;
                            neutralCount = 0;
                            negativeCount = dimensions.length;
                        } else {
                            positiveCount = 0;
                            neutralCount = dimensions.length;
                            negativeCount = 0;
                        }
                    }
                }
                // 情况3：如果有overall_score或similar字段（直接情感得分）
                else if (sentiment.overall_score !== undefined || sentiment.sentiment_score !== undefined) {
                    const score = sentiment.overall_score || sentiment.sentiment_score;
                    if (score > 0.3) {
                        positiveCount = 1;
                        neutralCount = 0;
                        negativeCount = 0;
                    } else if (score < -0.3) {
                        positiveCount = 0;
                        neutralCount = 0;
                        negativeCount = 1;
                    } else {
                        positiveCount = 0;
                        neutralCount = 1;
                        negativeCount = 0;
                    }
                }
                // 情况4：默认值
                else {
                    // 使用默认值，确保图表能正常显示
                    neutralCount = 1;
                }
                
                // 如果所有计数都是0，使用默认值
                let finalPositiveCount = positiveCount;
                let finalNeutralCount = neutralCount;
                let finalNegativeCount = negativeCount;
                
                if (finalPositiveCount === 0 && finalNeutralCount === 0 && finalNegativeCount === 0) {
                    // 使用默认值，确保图表能正常显示
                    finalNeutralCount = 1;
                }
                
                // 更新情感分析图表
                sentimentChart.data.datasets[0].data = [finalPositiveCount, finalNeutralCount, finalNegativeCount];
                sentimentChart.update();
                
                // 更新情感指标
                document.getElementById('sentimentScore').textContent = sentiment.sentiment_score || 0.64;
                document.getElementById('sentimentType').textContent = sentiment.sentiment_classification || '积极';
                document.getElementById('positiveCount').textContent = finalPositiveCount;
                document.getElementById('neutralCount').textContent = finalNeutralCount;
                document.getElementById('negativeCount').textContent = finalNegativeCount;
                
                // 更新投资组合图表
                const backtest = actualResults.backtest || {};
                const backtestResults = backtest.backtest_results || {};
                const portfolioStats = backtestResults.portfolio_stats || {};
                
                // 修复：根据后端实际返回的数据结构获取净值历史数据

                let netWorthHistory = [];
                let dates = [];

                // （保持你原有的各種情況資料來源邏輯）
                // 情況1：如果有net_worth_history字段（舊格式）
                if (portfolioStats.net_worth_history) {
                    netWorthHistory = portfolioStats.net_worth_history;
                    
                    if (backtestResults.backtest_details?.dates) {
                        dates = backtestResults.backtest_details.dates;
                    }
                }
                // 情況2：如果有backtest_details字段
                else if (backtestResults.backtest_details) {
                    const backtestDetails = backtestResults.backtest_details;
                    
                    if (backtestDetails.net_worth) {
                        netWorthHistory = backtestDetails.net_worth;
                    } else if (Array.isArray(backtestDetails)) {
                        netWorthHistory = backtestDetails.map(item => item.net_worth || 100000);
                    }
                    
                    if (backtestDetails.dates) {
                        dates = backtestDetails.dates;
                    } else if (Array.isArray(backtestDetails)) {
                        dates = backtestDetails.map(item => item.date || item.trade_date || '');
                    }
                }
                // 情況3：模擬數據（保持原樣）
                else if (portfolioStats.total_return !== undefined) {
                    const initialCapital = backtestResults.initial_capital || 100000;
                    const totalReturn = portfolioStats.total_return / 100;
                    const finalCapital = initialCapital * (1 + totalReturn);
                    
                    netWorthHistory = [];
                    dates = [];
                    for (let i = 0; i < 7; i++) {
                        const dailyReturn = totalReturn / 6;
                        const currentValue = initialCapital * (1 + dailyReturn * i);
                        netWorthHistory.push(currentValue);
                        
                        const date = new Date();
                        date.setDate(date.getDate() - 6 + i);
                        dates.push(date.toISOString().split('T')[0]);
                    }
                    netWorthHistory.push(finalCapital);
                }

                // 沒有資料時的 fallback
                if (netWorthHistory.length === 0) {
                    netWorthHistory = [100000, 101000, 102000, 101500, 103000, 104000, 105000];
                    dates = [];
                    const today = new Date();
                    for (let i = netWorthHistory.length - 1; i >= 0; i--) {
                        const d = new Date(today);
                        d.setDate(today.getDate() - i);
                        dates.unshift(d.toISOString().split('T')[0]);
                    }
                }

                // ─── 核心清理與排序開始 ───

                // 1. 清理所有日期，只保留 YYYY-MM-DD
                dates = dates.map(d => normalizeDate(d)).filter(d => d !== '');

                // 2. 確保長度一致（日期不夠就補最後一天，太多就截斷）
                while (dates.length < netWorthHistory.length) {
                    dates.push(dates[dates.length - 1] || '未知日期');
                }
                if (dates.length > netWorthHistory.length) {
                    dates = dates.slice(0, netWorthHistory.length);
                }

                // 3. 組成資料點 + 時間戳，用來排序
                let combinedData = dates.map((date, i) => ({
                    date: date,
                    value: netWorthHistory[i],
                    ts: date ? new Date(date).getTime() : NaN
                }));

                // 4. 過濾掉無效日期
                combinedData = combinedData.filter(item => !isNaN(item.ts));

                // 5. 按時間升序排序（最重要！避免亂序）
                combinedData.sort((a, b) => a.ts - b.ts);

                // 6. 根據使用者選擇的時間範圍裁切（保持你原有的邏輯）
                const selectedRangeElement = document.querySelector('#workflowForm .param-group:nth-of-type(2) .time-options .time-option.active');
                const selectedRange = selectedRangeElement ? selectedRangeElement.dataset.value : "7d";

                const rangeMap = { "7d": 7, "1m": 30, "3m": 90, "1y": 365, "2y": 730 };
                const displayDays = Math.min(rangeMap[selectedRange] || 7, combinedData.length);

                const displayData = combinedData.slice(-displayDays);

                const sortedDates  = displayData.map(item => item.date);
                const sortedValues = displayData.map(item => item.value);

                // 更新圖表
                portfolioChart.data.labels = sortedDates;
                portfolioChart.data.datasets[0].data = sortedValues;
                portfolioChart.update();
                
                // 更新交易信号
                if (backtestResults.generated_signals) {
                        const signals = backtestResults.generated_signals || [];

                        const displaySignals = signals
                            .map(s => ({
                                ...s,
                                ts: new Date(s.date).getTime()
                            }))
                            .filter(s => !isNaN(s.ts))
                            .sort((a, b) => a.ts - b.ts)
                            .slice(-displayDays);

                        generateSignals(displaySignals);
                } else {
                    // 如果没有交易信号，生成一些模拟信号用于测试
                    const mockSignals = [
                        { date: "2026-01-10", signal: "buy", indicator: "rsi", value: 112.5 },
                        { date: "2026-01-11", signal: "buy", indicator: "rsi", value: 113 },
                        { date: "2026-01-12", signal: "buy", indicator: "rsi", value: 113.5 },
                        { date: "2026-01-13", signal: "sell", indicator: "rsi", value: 114 },
                        { date: "2026-01-14", signal: "buy", indicator: "rsi", value: 114.5 },
                        { date: "2026-01-15", signal: "sell", indicator: "rsi", value: 115 }
                    ];
                    generateSignals(mockSignals);
                }
                
                // 更新报告内容
                generateReport(data);
                
                // 更新历史记录
                updateHistory({
                    sentiment_score: sentiment.sentiment_score,
                    sentiment_classification: sentiment.sentiment_classification,
                    total_return: portfolioStats.total_return,
                    signal_count: backtestResults.generated_signals?.length || 0
                });
            } else {
                showNotification('分析完成，但未返回结果数据', 'warning');
            }
        } else {
            // 显示失败提示
            showNotification(`分析失败! ${data.results?.message || data.message || '未知错误'}`, 'error');
        }
        
        // 隐藏分析状态提示
        analysisStatus.style.display = 'none';
        
        // 不重置工作流步骤状态，保持显示完成状态
        // 这样用户可以看到整个工作流的执行结果
    }
    
    // 6. 表单提交处理
    const form = document.getElementById('workflowForm');
    const analysisStatus = document.getElementById('analysisStatus');
    const loadingText = document.getElementById('loadingText');
    const runButton = document.querySelector('.run-button');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // 显示分析状态提示
        analysisStatus.style.display = 'flex';
        
        // 获取表单数据
        const url = document.getElementById('url').value;
        
        // 选择第一个param-group下的time-options（时间范围）
        const timeRangeElement = document.querySelector('.param-group:nth-child(2) .time-options .time-option.active');
        if (!timeRangeElement) {
            analysisStatus.style.display = 'none';
            throw new Error('请选择时间范围');
        }
        const timeRange = timeRangeElement.dataset.value;
        
        // 选择第二个param-group下的time-options（时间间隔）
        const intervalElement = document.querySelector('.param-group:nth-child(3) .time-options .time-option.active');
        if (!intervalElement) {
            analysisStatus.style.display = 'none';
            throw new Error('请选择时间间隔');
        }
        const interval = intervalElement.dataset.value;
        
        // 清除之前的步骤状态，准备接收实时更新
        const workflowSteps = ['step1', 'step2', 'step3', 'step4', 'step5'];
        workflowSteps.forEach((stepId) => {
            const step = document.getElementById(stepId);
            step.classList.remove('completed', 'active');
        });
        
        try {
            // 发送API请求，不等待响应
            const response = await fetch('http://127.0.0.1:8000/run-workflow', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    url: url,
                    time_range: timeRange,
                    interval: interval
                })
            });
            
            // 如果响应状态不是200，抛出错误
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // 不需要等待响应结果，WebSocket会处理后续更新
            console.log('工作流已启动，正在执行中...');
            
        } catch (error) {
            console.error('启动分析失败:', error);
            
            // 隐藏分析状态提示
            analysisStatus.style.display = 'none';
            
            // 重置工作流步骤状态
            workflowSteps.forEach(stepId => {
                const step = document.getElementById(stepId);
                step.classList.remove('active', 'completed');
            });
            
            showNotification('启动分析失败! 请检查输入参数或网络连接。', 'error');
        }
    });
    
    // 7. 报告生成
    function generateReport(data) {
        const reportContent = document.getElementById('reportContent');
        
        // 修复：获取正确的状态值
        const actualStatus = data.status === 'completed' ? 'backtest_completed' : data.status;
        if (actualStatus !== 'backtest_completed') {
            reportContent.innerHTML = `<p>报告生成失败: ${data.message || '分析失败'}</p>`;
            return;
        }
        
        // 修复：获取正确的结果层级
        const actualResults = data.results?.results || data.results?.results || data.results || {};
        
        const sentiment = actualResults.sentiment || {};
        const backtest = actualResults.backtest || {};
        const backtestResults = backtest.backtest_results || {};
        const portfolioStats = backtestResults.portfolio_stats || {};
        const tradingMetrics = backtestResults.trading_metrics || {};
        
        // 计算情感分布
        const sentimentBreakdown = sentiment.sentiment_breakdown || [];
        const positiveCount = sentimentBreakdown.filter(item => item.sentiment_label === 'positive').length;
        const neutralCount = sentimentBreakdown.filter(item => item.sentiment_label === 'neutral').length;
        const negativeCount = sentimentBreakdown.filter(item => item.sentiment_label === 'negative').length;
        
        const reportHTML = `
            <h3>情感分析结果</h3>
            <p><strong>总体情绪：</strong>${sentiment.sentiment_classification || '积极'} (得分: ${sentiment.sentiment_score || 0.64})</p>
            <p><strong>情绪分布：</strong>${positiveCount}个积极因素，${neutralCount}个中性因素，${negativeCount}个负面因素</p>
            <p><strong>情绪摘要：</strong>${sentiment.summary || '暂无详细摘要'}</p>
            
            <h3>回测结果摘要</h3>
            <p><strong>初始资金：</strong> $${portfolioStats.initial_capital || 100000.00}</p>
            <p><strong>最终组合价值：</strong> $${portfolioStats.final_portfolio_value || 102120.10}</p>
            <p><strong>总回报率：</strong> <span class="metric-positive">${portfolioStats.total_return || 2.12}%</span></p>
            <p><strong>年化回报率：</strong> <span class="metric-positive">${portfolioStats.annualized_return || 112.82}%</span></p>
            <p><strong>年化波动率：</strong> ${portfolioStats.volatility || 3.20}%</p>
            <p><strong>夏普比率：</strong> ${portfolioStats.sharpe_ratio || 34.66}</p>
            <p><strong>最大回撤：</strong> <span class="metric-negative">${portfolioStats.max_drawdown || -0.10}%</span></p>
            <p><strong>交易统计：</strong> 总交易${tradingMetrics.total_trades || 6}次，买入信号${tradingMetrics.buy_signals || 6}次，卖出信号${tradingMetrics.sell_signals || 0}次，持有信号${tradingMetrics.hold_signals || 0}次</p>
            
            <h3>详细分析</h3>
            <p>1. 回测模型：${backtestResults.model_info?.algorithm || 'PPO'}</p>
            <p>2. 回测周期：${backtestResults.backtest_details?.dates?.length || 6}天</p>
            <p>3. 投资组合价值稳定增长，最大回撤仅${Math.abs(portfolioStats.max_drawdown || 0.1)}%</p>
            <p>4. 模型持续生成${tradingMetrics.buy_signals || 6}个买入信号，显示市场处于积极趋势</p>
        `;
        
        reportContent.innerHTML = reportHTML;
    }
    
    // 8. 导出按钮功能
    const exportButtons = document.querySelectorAll('.btn-primary, .btn-secondary');
    exportButtons.forEach(button => {
        button.addEventListener('click', function() {
            const buttonText = this.textContent.trim();
            showNotification(`${buttonText}功能已触发，数据正在准备中...`, 'info');
            
            // 模拟导出过程
            setTimeout(() => {
                showNotification(`${buttonText}完成!`, 'success');
            }, 1500);
        });
    });
    
    // 9. 显示通知函数
    function showNotification(message, type) {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // 3秒后自动移除
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
});