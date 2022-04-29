$(function(){
    let btn = document.querySelector('button[name=index]')
    if (location.href.includes('index.html')){
        btn.style.borderBottom = '3px solid #fff'
        btn.style.color = '#fff'
    }else if(location.href.includes('demo.html')){
        btn = document.querySelector('button[name=demo]')
        btn.style.borderBottom = '3px solid #fff'
        btn.style.color = '#fff'
    }else if(location.href.includes('contact.html')){
        btn = document.querySelector('button[name=contact]')
        btn.style.borderBottom = '3px solid #fff'
        btn.style.color = '#fff'
    }else if(location.href.includes('documentation.html')){
        btn = document.querySelector('button[name=documentation]')
        btn.style.borderBottom = '3px solid #fff'
        btn.style.color = '#fff'
    }
});
function navBarClick(event){
    let name = event.name
    if (name!=='index')
        location.href="./"+name+".html";
    else location.href = './index.html';
}