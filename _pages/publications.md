---
layout: page
permalink: /publications/
title: Publications
description: Publications are listed in reversed chronological order.
years: [2025,2024,2023,2022,2021,2020,2019,2018,2017]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">
<nav id="year-nav" class="navbar fixed-bottom container" style="margin-bottom: -50px; align-self: center;">
  <p class="post-description" style="padding-bottom: 15px; align-self: center"> Jump to:
  {%- for y in page.years %}
      <a href="#year-{{y}}" class="btn btn-sm z-depth-0" style="padding: 0 0 0 0" role="button">{{y}}</a>
  {% endfor %}
  </p>
</nav>

{%- for y in page.years %}
  <h2 class="year" id="year-{{y}}">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
