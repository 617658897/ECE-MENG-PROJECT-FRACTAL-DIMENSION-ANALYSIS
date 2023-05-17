import React, { Component } from 'react';
import './App.css';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Button from 'react-bootstrap/Button';

import 'bootstrap/dist/css/bootstrap.css';

class App extends Component {

  constructor(props) {
    super(props);

    this.state = {
      isUploaded: false,
      isLoading: false,
      formData: {
        fileUploadState: '',
        img: '',
        selectedImg: null,
      },
      result: ''
    };

    this.inputReference = React.createRef();
  }

  handleChange = (event) => {
    const value = event.target.value;
    const name = event.target.name;
    var formData = this.state.formData;
    formData[name] = value;
    this.setState({
      formData
    });
  }

  handlePredictClick = (event) => {
    const img = this.state.img;
    this.setState({ isLoading: true });
    fetch('http://127.0.0.1:5000/prediction/',
      {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify(img)
      })
      .then(response => response.json())
      .then(response => {
        this.setState({
          result: response.result,
          isLoading: false
        });
      });

  }

  handleCancelClick = (event) => {
    this.setState({
      result: "",
      isUploaded: false,
      fileUploadState: '',
      img: '',
      selectedImg: null,
    });
    const myNode = document.getElementById("resultUpload");
    myNode.innerHTML = '';
    document.getElementById("imagePath").innerHTML = '';
  }

  fileUploadAction = (event) => {
    this.inputReference.current.click();
    this.setState({ isUploaded: true })
  }

  fileUploadInputChange = (e) => {
    this.setState({ fileUploadState: e.target.value })
    const file = e.target.files[0];
    this.setState({ 
      selectedImg : URL.createObjectURL(file)
    })
    const that = this
    const reader = new FileReader();
    
    reader.onloadend = (e) => {

      that.setState({
        img: e.target.result
      })
      var fig = document.createElement("img");
      fig.src = e.target.result;
      fig.style.marginBottom = "20px";
      document.getElementById("resultUpload").appendChild(fig);
      document.getElementById("imagePath").innerHTML = `
  <span style="color: black;">File path: ${this.state.fileUploadState}</span>
`;

      //console.log(that.state.img)

    }
    reader.readAsDataURL(file)
    //reader.readAsText(file);
    //console.log("ll")
    //console.log(this.state.img)

  }

  render() {
    const isLoading = this.state.isLoading;
    const formData = this.state.formData;
    const result = this.state.result;
    const fileUploadState = this.state.fileUploadState;
    const isUploaded = this.state.isUploaded;
    const img = this.state.img;
    const selectedImg = this.state.selectedImg;





    return (
      <Container>
        <div>
          <h1 className="title">Fractal Dimension Analysis</h1>
        </div>
        <div className="content">
          <Form>
            <Row>
              <Col>
                <Form.Label>Select the image: </Form.Label>
                <Form.Label>(Only files with the following extensions are allowed: png, gif, jpg, jpeg)</Form.Label>

                <div className='m-3'>
                  <input type="file" hidden ref={this.inputReference} accept="image/*" onChange={this.fileUploadInputChange.bind(this)} />
                  <Button className="ui button" onClick={this.fileUploadAction} variant="outline-secondary">
                    Image Upload
                  </Button>
                </div>
              </Col>

              <Col>
                <div id="imagePath"></div>
                <div id="resultUpload"></div>
              </Col>

            </Row>


            <Row>
              <Col>
                <Button
                  block
                  variant="success"
                  disabled={isLoading}
                  onClick={!isLoading ? this.handlePredictClick : null}>
                  {isLoading ? 'Making prediction' : 'Predict'}
                </Button>
              </Col>
              <Col>
                <Button
                  block
                  variant="danger"
                  disabled={isLoading}
                  onClick={this.handleCancelClick}>
                  Reset prediction
                </Button>
              </Col>
            </Row>
          </Form>
          {result === "" ? null :
            (<Row>
              <Col className="result-container">
                <img className="result-image" src={result} />
              </Col>
            </Row>)
          }
        </div>
      </Container >
    );
  }
}

export default App;