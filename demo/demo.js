//trim function
if (!String.prototype.trim) {
    String.prototype.trim = function () {
        return this.replace(/^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, '');
    };
}
// string이 json형식인지 확인하는 함수
function isJsonString(str) {
    try{
        JSON.parse(str);
    }catch (e){
        return false;
    }
    return true;
}
//Basic Rendering
let sampleA, sampleB;
fetch('./src/test_A.json')
    .then(response => {
        return response.json();
    })
    .then(jsondata => sampleA = jsondata);
fetch('./src/test_B.json')
    .then(response => {
        return response.json();
    })
    .then(jsondata => sampleB = jsondata);
$(function(){
    // Anchor Submit Action
    $('#ajax-submit').on('click', function(event){
        let radio = document.querySelectorAll('.upload-opt');
        let upload_type;
        radio.forEach((node) => node.checked ? upload_type = node.value : null)
        if (upload_type ==='File') {
            if (document.getElementById("upload-file").value.length !== 0) {
                document.querySelector('.mdl-js-spinner').style.visibility = 'visible'
                setTimeout(anchorSubmit, 10)
            } else {
                alert("Please submit the file.")
                document.querySelector('.upload-file-button').focus()
            }
        }else {
            let text = document.getElementsByClassName('text-section')[0].value;
            text == null ?  text = "" : text.trim();
            if (!isJsonString(text)){
                alert("Please use JSON format.");
                document.querySelector('.text-section').focus()
            }
            else if (text.length!==0){
                document.querySelector('.mdl-js-spinner').style.visibility = 'visible'
                setTimeout(TextAnchorSubmit(text), 10)
            } else {
                alert("Please fill the text box.")
                document.querySelector('.text-section').focus()
            }
        }
        return false
    })
    // Scaler Submit Action
    $('.scaler-data-submit-btn').on('click', function(event){
        ScalerSubmit();
    })
    // Default Anchor Chart
    anchorChart = new Chart(document.getElementById('anchor-result-chart').getContext('2d'), {
        type: 'bar',
        data: anchorDefaultData,
        options: anchorDefaultOption
    })
    // Default Scaler Chart
    scalerChart = new Chart(document.getElementById('scaler-result-chart').getContext('2d'), {
        type: 'line',
        data: scalerDefaultData,
        options: scalerDefaultOption
    })
    // Scaler Target Size Change
    $('.scalar-radio').on('click', function(event){
        let latencyUnit = event.target.value;
        let minNotice = document.querySelector('#min-notice');
        let maxNotice = document.querySelector('#max-notice');
        let rangeNotice = document.querySelector('#scaler-size-range');
        if (latencyUnit === 'ps'){
            minNotice.innerHTML = '(32x32x3 pixel size)'
            maxNotice.innerHTML = '(256x256x3 pixel size)'
            rangeNotice.setAttribute('min', '32')
        }else{
            minNotice.innerHTML = '(16 Batchsize)'
            maxNotice.innerHTML = '(256 Batchsize)'
            rangeNotice.setAttribute('min', '16')
        }
    })
    // Scaler Range Slider
    let rangeSlider = document.getElementById("scaler-size-range");
    let rangeSliderSize = document.getElementById("scaler-size-value");
    let rangeSliderLatency = document.getElementById("scaler-size-latency");
    let minRangeSliderValue
    rangeSliderSize.innerHTML = rangeSlider.value;
    rangeSlider.oninput = function() {
        rangeSliderSize.innerHTML = this.value
        minRangeSliderValue = rangeSlider.getAttribute('min')
        if (Array.isArray(predLatency) && predLatency.length){
            rangeSliderLatency.innerHTML = predLatency[this.value - minRangeSliderValue]
        }else{
            rangeSliderLatency.innerHTML = 0
        }
    }
    $('.upload-opt').on('click', function (event){
        let uploadUnit = event.target.value;
        let uploadDiv = document.querySelector('.file-upload-section');
        let textDiv = document.querySelector('.text-section');
        if (uploadUnit==="Text"){
            uploadDiv.style = 'visibility: hidden';
            textDiv.style = 'display: block';
            document.querySelector('#ajax-upload-form').reset();
            // document.querySelector('#ajax-upload-form').value = "";
            document.querySelector('#upload-text').value = "upload file";
        }
        else {
            uploadDiv.style = 'display: flex';
            textDiv.style = 'display: none';
            document.querySelector('.text-section').value = '';
        }
    })
});
function download(testfile){ //axios 사용해서 anchor instance Test Json File direct download 하는 함수
    if(testfile == 'test-file-a'){
        url_link = 'https://raw.githubusercontent.com/anonymous-profet/profet/main/demo/src/test_A.json'
        download_file_name = 'test_A.json'}
    else{
        url_link = 'https://raw.githubusercontent.com/anonymous-profet/profet/main/demo/src/test_B.json'
        download_file_name = 'test_B.json'}

    axios({
        url: url_link,
        method:'GET',
        responseType: 'blob'
    })
        .then((response) => {
            const url = window.URL.createObjectURL(new Blob([response.data]))
            const link = document.createElement('a')
            link.href = url
            link.setAttribute('download', download_file_name)
            document.body.appendChild(link)
            link.click()
        })
}
function sampleData(testfile){
    let textBox = document.querySelector('.text-section');
    let anchorInput = document.querySelector('#anchor-latency');
    if(testfile == 'test-file-a'){
        const str = JSON.stringify(sampleA);
        textBox.value = str;
        anchorInput.value = 122400;
        // const bytes = new TextEncoder().encode(str);
        // const blob = new Blob([bytes], {
        //     type: "application/json;charset=utf-8"
        // });
        // let sample = new File([blob], 'test_A.json',{type:"application/json"});
        // formData.append('upload-file', sample);
        // formData.append('anchor_instance', 'g3s_xlarge');
        // formData.append('anchor_latency', 122400);

    }
    else {
        const str = JSON.stringify(sampleB);
        textBox.value = str;
        anchorInput.value = 1529870;
        // const str = JSON.stringify(sampleB);
        // const bytes = new TextEncoder().encode(str);
        // const blob = new Blob([bytes], {
        //     type: "application/json;charset=utf-8"
        // });
        // let sample = new File([blob], 'test_B.json',{type:"application/json"});
        // formData.append('upload-file', sample);
        // formData.append('anchor_instance', 'g3s_xlarge');
        // formData.append('anchor_latency', 1529870);
    }
    // $.ajax({
    //     type: "POST",
    //     url: 'https://vxummx2cra.execute-api.us-east-1.amazonaws.com/profet-dev/predict',
    //     header: {
    //         "Content-Type": "application/json",	//Content-Type 설정
    //         "X-HTTP-Method-Override": "POST",
    //         'Access-Control-Allow-Origin': '*',
    //         'Access-Control-Allow-Methods': 'POST',
    //         'Access-Control-Allow-Headers': '*',
    //         'Access-Control-Allow-Age': 3600
    //     },
    //     dataType: "json",
    //     data: formData,
    //     AccessControlAllowOrigin: '*',
    //     crossDomain: true,
    //     contentType: false,
    //     async: false,
    //     enctype: 'multipart/form-data',
    //     processData: false,
    //     success: function (data, status) {
    //         let result = data['body']
    //         result = result.replace("\"", "");
    //         result = result.split('&');
    //         for (let j = 0; j < 3; j++) {
    //             instanceList[result[j + 3]].latency = Math.round(result[j] * 100) / 100;
    //         }
    //         showAnchorResult();
    //         document.querySelector('.mdl-js-spinner').style.visibility = 'hidden'
    //     },
    //     error: function (e) {
    //         console.log("ERROR : ", e);
    //         $("#btnSubmit").prop("disabled", false);
    //         alert(e.message);
    //     }
    // });
}
// Anchor Prediction 을 위해 ajax 를 통해 File 을 업로드 하는 함수
// anchorChart 함수를 호출하여 Chart 를 Rendering 함
// radio 버튼의 값에 따라 우선순위
function anchorSubmit(){
    let form = $("#ajax-upload-form");
    let formData = new FormData(form[0]);
    formData.append('anchor_instance', $("#anchor-instance").val());
    formData.append('anchor_latency', $("#anchor-latency").val());
    $.ajax({
    type: "POST",
    url: 'https://vxummx2cra.execute-api.us-east-1.amazonaws.com/profet-dev/predict',
    header: {
        "Content-Type": "application/json",	//Content-Type 설정
        "X-HTTP-Method-Override": "POST",
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Age': 3600
    },
    dataType: "json",
    data: formData,
    AccessControlAllowOrigin: '*',
    crossDomain: true,
    contentType: false,
    async: false,
    enctype: 'multipart/form-data',
    processData: false,
    success: function (data, status) {
        let result = data['body']
        result = result.replace("\"", "");
        result = result.split('&');
        for (let j = 0; j < 3; j++) {
            instanceList[result[j + 3]].latency = Math.round(result[j] * 100) / 100;
        }
        showAnchorResult();
        document.querySelector('.mdl-js-spinner').style.visibility = 'hidden';
        formData.forEach(e => console.log(e));
    },
    error: function (e) {
        console.log("ERROR : ", e);
        $("#btnSubmit").prop("disabled", false);
        alert(e.message);
    }
    });
}
//text값을 파일로 만들어서 값을 전달
function TextAnchorSubmit(data) {
    //text를 파일 형태로 생성
    let formData = new FormData();
    const bytes = new TextEncoder().encode(data);
    const blob = new Blob([bytes], {
        type: "application/json;charset=utf-8"
    });
    let sample = new File([blob], 'textData.json',{type:"application/json"});
    formData.append('upload-file', sample);
    formData.append('anchor_instance', $("#anchor-instance").val());
    formData.append('anchor_latency', $("#anchor-latency").val());
    $.ajax({
        type: "POST",
        url: 'https://vxummx2cra.execute-api.us-east-1.amazonaws.com/profet-dev/predict',
        header: {
            "Content-Type": "application/json",	//Content-Type 설정
            "X-HTTP-Method-Override": "POST",
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Age': 3600
        },
        dataType: "json",
        data: formData,
        AccessControlAllowOrigin: '*',
        crossDomain: true,
        contentType: false,
        async: false,
        enctype: 'multipart/form-data',
        processData: false,
        success: function (data, status) {
            let result = data['body']
            result = result.replace("\"", "");
            result = result.split('&');
            for (let j = 0; j < 3; j++) {
                instanceList[result[j + 3]].latency = Math.round(result[j] * 100) / 100;
            }
            showAnchorResult();
            document.querySelector('.mdl-js-spinner').style.visibility = 'hidden'
        },
        error: function (e) {
            console.log("ERROR : ", e);
            $("#btnSubmit").prop("disabled", false);
            alert(e.message);
        }
    });
}

// Anchor Chart 를 그리는 함수
//instacneList의 값 중 cost-> price로 변경하였고, 원래 total로 값이 추가 되던 변수명을 cost로 변경하였습니다.
let instanceList = {'g3s_xlarge': {price: 0.75, latency: 0}, 'g4dn_xlarge': {price: 0.526, latency: 0}, 'p2_xlarge': {price: 0.9, latency: 0}, 'p3_2xlarge': {price: 3.06, latency: 0}}
let anchorChart
function showAnchorResult(){
    if(typeof anchorChart != 'undefined'){
        anchorChart.destroy()
    }

    // Anchor Result Table 에 값을 업데이트
    const anchorResult = document.querySelectorAll('.result-table-element')
    for (let i = 0; i < anchorResult.length; i++) {
        anchorResult[i].style.display = 'table-row'
        anchorResult[i].style.color = 'rgba(0,0,0,.87)'
    }
    const anchorInstance = $('#anchor-instance option:selected').val()
    document.getElementById(anchorInstance).style.display = 'none'  //선택된 instance 제외하고 보여주도록

    for (let key in instanceList){
        let predInstance = instanceList[key];
        instanceList[key].cost = Math.round((predInstance.price * predInstance.latency)/36)/100
    }
    for (let key in instanceList) {
        if (key !== anchorInstance) {
            let predInstance = instanceList[key]
            for (let i in predInstance) {
                if (i !== 'price') {
                    document.querySelector('#' + key + ' .' + i).innerHTML = i==='cost'?predInstance[i]:Math.round(predInstance[i]).toString()
                }
            }
        }
    }

    // Anchor Result Chart 그리는 코드
    let instanceLabel = []
    for (let name in instanceList) {
        name !== anchorInstance ? instanceLabel.push(name.toString()) : null
    }
    let instanceCost = []
    for (let i = 0; i < instanceLabel.length; i++) {
        instanceCost.push(instanceList[instanceLabel[i]].cost)
    }
    let instanceLatency = []
    for (let i = 0; i < instanceLabel.length; i++) {
        instanceLatency.push(Math.round(instanceList[instanceLabel[i]].latency))
    }
    anchorChart = new Chart(document.getElementById('anchor-result-chart').getContext('2d'), {
        type: 'bar',
        data: anchorData(instanceLabel, instanceLatency, instanceCost),
        options: anchorOption
    })
}

let scalerChart
let predLatency
let chartSizeList = {"bs": [16, 32, 64, 128, 256], "ps": [32, 64, 128, 224, 256]}
let sliderSizeList = {"bs" : Array(256 - 16 + 1).fill().map((_, idx) => 16 + idx), "ps" : Array(256 - 32 + 1).fill().map((_, idx) => 16 + idx)}
let scalerWeight = {"bs" : {'g3s_xlarge':[-0.06780512, 4.24740752e-03, -3.03861896e-07],'g4dn_xlarge':[-0.07318534, 4.44307450e-03, -9.78530747e-07],'p2_xlarge':[-0.06536447, 4.10525099e-03, 2.18967283e-07],'p3_2xlarge':[-0.07426024, 4.65321010e-03, -1.79507257e-06]},
                    "ps" : {'g3s_xlarge':[-0.0025238, -4.27576141e-04, 1.69900581e-05],'g4dn_xlarge':[-0.01299441, -1.44524706e-04, 1.60282082e-05],'p2_xlarge':[-0.01437971, -6.69419624e-05, 1.58622467e-05],'p3_2xlarge':[-0.00843561, -4.57321020e-04, 1.73672402e-05]}}

function ScalerSubmit(){
    if (typeof scalerChart != 'undefined'){
        scalerChart.destroy();
    }
    let targetInstance = $("#target-instance").val()
    const sizeTypeList = document.querySelectorAll('.select-scaler-size-type label input')
    let bs_ps = sizeTypeList[0].checked?"bs":"ps"
    let latencyMin = Number(document.querySelector('#latency-min').value)
    let latencyMax = Number(document.querySelector('#latency-max').value)
    if (latencyMin===0||latencyMax===0){
        alert('please fill latency min/max')
    }else {
        let weight = scalerWeight[bs_ps][targetInstance]
        let chartSize = chartSizeList[bs_ps]
        let scaledLatency = sliderSizeList[bs_ps].map(function(x) { return 1 * weight[0] + x * weight[1] + (x**2) * weight[2]})
        predLatency = scaledLatency.map(function(x) { return (x * (latencyMax - latencyMin) + latencyMin).toFixed(4)})

        predLatency[0] = latencyMin
        predLatency[predLatency.length - 1] = latencyMax
        
        let chartPredLatency = chartSize.map(function(x) { return predLatency[x - Math.min.apply(Math, chartSize)]})

        scalerChart = new Chart(document.getElementById('scaler-result-chart').getContext('2d'), {
            type: 'line',
            data: scalerData(chartSize, chartPredLatency)
        })
    }
}
window.addEventListener('resize', function(event) {
    let width_val = document.querySelector('.anchor-data-layout').clientWidth
    document.querySelector('.scaler-data-layout').style.width = (width_val-40).toString().concat('px')
}, true);
document.addEventListener("DOMContentLoaded", function() {
    let width_val = document.querySelector('.anchor-data-layout').clientWidth
    document.querySelector('.scaler-data-layout').style.width = (width_val-40).toString().concat('px')
})
let loadFile = function(event) {
    let output = document.getElementById('output');
    let i = event.value.split('fakepath\\')[1]
    document.querySelector('#upload-text').value = i;
};

function createDiagonalPattern(color = 'black') {
    let shape = document.createElement('canvas')
    shape.width = 10
    shape.height = 10
    let c = shape.getContext('2d')
    c.strokeStyle = 'white'
    c.fillStyle = color
    c.fillRect(0, 0, shape.width, shape.height);
    c.beginPath()
    c.moveTo(2, 0)
    c.lineTo(10, 8)
    c.stroke()
    c.beginPath()
    c.moveTo(0, 8)
    c.lineTo(2, 10)
    c.stroke()
    return c.createPattern(shape, 'repeat')
}

let anchorDefaultData = {
    labels: ['A', 'B', 'C'],
    datasets: [{
        label: 'Latency',
        yAxesGroup: 'latencyGroup',
        backgroundColor: [createDiagonalPattern("#3e95cd"), createDiagonalPattern("#ffa400"), createDiagonalPattern("#3cba9f")],
        data: [],
        borderColor: ["#3e95cd","#ffa400","#3cba9f"],
        borderWidth: 2
    }, {
        label: 'Cost',
        yAxesGroup: 'costGroup',
        backgroundColor: ["#a5d4ff", "#ffd1a3", "#cdfcf3"],
        data: []
    }]
};

let anchorDefaultOption = {
    responsive:true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            labels: {
                generateLabels: function(chart) {
                    const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                    for (let key in labels) {
                        labels[key].fillStyle  = key==='0'?createDiagonalPattern('#A2A2A2'):"#A2A2A2";
                        labels[key].strokeStyle = "#A2A2A2";
                    }
                    return labels;
                },
            }
        },
    },
    scales: {
        x: {
            title: {
                display: true,
                text: 'Target Instance Type',
                font : {
                    size : 14,
                }
            }
        },
        y: {
            type: 'linear',
            position: 'left',
            scalePositionLeft: true,
            title: {
                display: true,
                text: 'Latency (us)',
                font : {
                    size : 14,
                }
            },
            min: 0,
            max: 10,
            grid: {
                display: false
            }
        },
        y1: {
            type: 'linear',
            position: 'right',
            scalePositionLeft: false,
            title: {
                display: true,
                text: 'Cost',
                font : {
                    size : 14,
                }
            },
            min: 0,
            max: 10,
            grid: {
                display: false
            }
        }
    },
};

let scalerDefaultData = {
    labels: [16, 32, 64, 128, 256],
    datasets: [{
        label: 'scaler',
        data: [160, 320, 640, 1280, 2560],
        tension: 0.4
    }]
}

let scalerDefaultOption = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
        x: {
            title: {
                display: true,
                text: 'Target Size',
                font : {
                    size : 14,
                }
            }
        },
        y: {
            title: {
                display: true,
                text: 'Latency (us)',
                font : {
                    size : 14,
                }
            }
        }
    },
    plugins : {
        legend : {
            display : false,
        }
    }
}

function anchorData(instanceLabel, instanceLatency, instanceCost){
    return {
        labels: instanceLabel,
        datasets: [
            {
                label: "latency",
                backgroundColor: [createDiagonalPattern("#3e95cd"), createDiagonalPattern("#ffa400"), createDiagonalPattern("#3cba9f")],
                data: instanceLatency,
                yAxisID: 'latency-y',
                borderColor: ["#3e95cd","#ffa400","#3cba9f"],
                borderWidth: 2,
            },{
                label: "cost",
                backgroundColor: ["#a5d4ff", "#ffd1a3", "#cdfcf3"],
                data: instanceCost,
                yAxisID: 'cost-y',
            }
        ]
    }
};

let anchorOption = {
    responsive:true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            labels: {
                generateLabels: function(chart) {
                    const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                    for (let key in labels) {
                        labels[key].fillStyle  = key==='0'?createDiagonalPattern('#A2A2A2'):"#A2A2A2";
                        labels[key].strokeStyle = "#A2A2A2";
                    }
                    return labels;
                },
            }
        },
    },
    scales: {
        x: {
            title: {
                display: true,
                text: 'Target Instance Type',
            }
        },
        'latency-y': {
            type: 'linear',
            position: 'left',
            scalePositionLeft: true,
            title: {
                display: true,
                text: 'Latency (us)',
                font : {
                    size : 14,
                }
            },
            ticks: {
                callback: (val) => (val.toExponential())
            },
            min: 0,
            grid: {
                display: false
            }
        },
        'cost-y': {
            type: 'linear',
            position: 'right',
            scalePositionLeft: false,
            title: {
                display: true,
                text: 'Cost',
                font : {
                    size : 14,
                }
            },
            min: 0,
            grid: {
                display: false
            }
        }
    }
};

function scalerData(sizeLabel, predLatency){
    return {
        labels: sizeLabel,
        datasets: [
            {
                type: 'line',
                label: 'Line Dataset',
                data: predLatency,
                tension: 0.4
            }
        ]
    }
};

let scalerOption = {
    responsive:true,
    maintainAspectRatio: false,
    scales: {
        x: {
            title: {
                display: true,
                text: 'Target Size',
                font : {
                    size : 14,
                }
            }
        },
        y: {
            title: {
                display: true,
                text: 'Latency (us)',
                font : {
                    size : 14,
                }
            }
        }
    }
}
