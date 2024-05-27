import streamlit as st
from src.sidebar import sidebar


sidebar(auto_expand=True)

st.markdown(
    """
<style>
.block-container:not(.st-emotion-cache-1v7bkj4){
	max-width: 95%;
	background-color: #568B97;
	padding: 45px 30px 15px 0px !important;
}

.stMarkdown{
	text-align: center;
}

.root-div {
	display: flex;
	align-items: center;
	height: 100%;
}
.left-div {
	width: 40%;
	display: inline-block;
	overflow: hidden;
}
.left-div > img {
	width:200px;
    height:100px;
}
.right-div {
	width: 60%;
}
.divider {
	width: 80%;
	height: 2px;
	background-color: white;
	margin: 2em 0em;
	display: inline-block;
}
.main-title{
	font-weight: 900;
	font-size: 70px;
	text-align: center;
	color: white;
	margin-top: 2em;
	margin-bottom: 0.5em;
}
.promotion {
	text-align: center;
	color: white;
}
.names {
	text-align: center;
	color: white;
	font-weight: 800;
}
.unsplash-link {
	color: white !important;
}
.unsplash-attribution{
	margin-top: 1em;
	color: white;
}
.launch-paragraph {
	margin-top: 4.5em;
}
.launch-button {
	display: inline-block ;
	padding: 0.4em 1em !important;
	background-color: white;
	color:#568B97 !important;
	text-decoration: none !important;
	font-weight: 900;
	border-radius: 8px;
	opacity: 0.9;
}
.launch-button:hover{
	opacity : 1;
	color: #ef3b6e !important;
}
.cover-image{
	position: relative;
	left: -2em;
}
</style>

<div class="root-div">
	<div class="left-div">
		 <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Logo_rakuten.jpg" alt="Image de courverture" height=100% class="cover-image">
	</div>
	<div class="right-div">
		<h1 class="main-title">Challenge Rakuten</h1>
		<div class="divider"></div>
		<h4 class="promotion">Promotion Data Scientist Bootcamp Septembre 2023</h4>
		<h3 class="names">M'BALI Guillaume</h3>
		<p class="launch-paragraph"><a class="launch-button" href="Introduction" target="_self">Démarrer &#128640;</a></p>
		<p class="unsplash-attribution">
			Photo by <a href="https://unsplash.com/@kmuza" class="unsplash-link">Carlos Muza</a> on <a href="https://unsplash.com/photos/hpjSkU2UYSU" class="unsplash-link">Unsplash</a>
  		</p>
	</div>
</div>

""",
    unsafe_allow_html=True,
)
