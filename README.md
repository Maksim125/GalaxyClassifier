# GalaxyClassifier

## About the project

This is a neural network that classifies galaxy clusters based on the images you feed it, and a web app that goes along with it that lets you upload and classify images in your browser.

<i>The website is not live because maintaining an entire tensorflow environment on a cloud platform gets expensive, and this project does not have the return on investment for that to be a sound expense.</i>

## Overview

Startup the webpage on local host, and upload an image! You'll get the CNN's prediction of what kind of galaxy cluster it is, and a brief description of that galaxy type.

## Setup

Running the webpage on local host is as simple as:

    pipenv --python 3.7
    pipenv install -r requirements.txt
    python setup.py

