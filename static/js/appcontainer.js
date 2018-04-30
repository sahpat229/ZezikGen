import React from 'react';
import Slider from 'material-ui/Slider';
import TextField from 'material-ui/TextField';

export default class AppContainer extends React.Component {
    constructor(props) {
        super(props);
        this.handleSeedChange = this.handleSeedChange.bind(this);
        this.handleTemperatureChange = this.handleTemperatureChange.bind(this);
        this.handleOutputLengthChange = this.handleOutputLengthChange.bind(this);
        this.state = {
            primer: '',
            temperature: 1.0,
            n_chars_generate: 600,
            loading: false,
            generated_sample: {}
        };
    }

    handleSeedChange(value) {
        this.setState({primer: value});
    }

    handleTemperatureChange(value) {
        this.setState({temperature: value});
    }

    handleOutputLengthChange(value) {
        this.setState({n_chars_generate: value});
    }

    generateText(dataset) {
            $.ajax({
                type: 'POST',
                url: '/generate_text',
                data: {
                    dataset: dataset,
                    primer: this.state.primer,
                    temperature: this.state.temperature,
                    n_chars_generate: this.state.n_chars_generate
                },
                success: (d) => {
                    this.setState({
                        loading: false,
                        generated_sample: JSON.parse(d)
                    }, () => {return console.log(this.state)})
                }
            });
    }

    renderSample() {
        let loaded_sample = this.state.generated_sample;
        let display_name = null;
        switch (loaded_sample.dataset) {
            case 'zizek':
                display_name = 'Slavoj Žižek';
            case 'shakespeare':
                display_name = 'William Shakespeare';
            case 'graham'
                display_name = 'Paul Graham';
            default:
                display_name = '';
}
    }

    render() {
        return (
            <div>
                <header></header>
                <main role="main">
                    <div className="container">
                        <h1 className="display-4">PURE IDEOLOGY GENERATOR</h1>
                        <div className="row">
                            <div className="col-md-4">
                            
                                SEED TEXT
                                <div className="row">
                                    <div className="col-12">
                                        <TextField
                                            id="field-primer:"
                                            type="text"
                                            fullWidth={true}
                                            value={this.state.primer}
                                            onChange={(e) => {this.handleSeedChange(e.target.value)}}
                                        />
                                    </div>
                                </div>
                                
                                LSTM TEMPERATURE
                                <div className="row">
                                    <div className="col-8">
                                        <Slider
                                            min={.01}
                                            max={1.00}
                                            step={.01}
                                            style={{margin: 0}}
                                            onChange={(e, v) => {this.handleTemperatureChange(v)}}
                                            value={this.state.temperature}
                                        />
                                    </div>
                                    <div className="col-4">
                                        <TextField
                                            id="field-temperature"
                                            type="number"
                                            min={.01}
                                            max={1.00}
                                            step={.01}
                                            fullWidth={true}
                                            value={this.state.temperature}
                                            onChange={(e) => {this.handleTemperatureChange(e.target.value)}}
                                        />
                                    </div>
                                </div>
                                
                                # CHARACTERS TO GENERATE
                                <div className="row">
                                    <div className="col-8">
                                        <Slider
                                            min={1}
                                            max={1000}
                                            step={1}
                                            onChange={(e, v) => {this.handleOutputLengthChange(v)}}
                                            value={this.state.n_chars_generate}
                                        />
                                    </div>
                                    <div className="col-4">
                                        <TextField
                                            id="field-chars"
                                            type="number"
                                            min={1}
                                            max={1000}
                                            step={1}
                                            fullWidth={true}
                                            value={this.state.n_chars_generate}
                                            onChange={(e) => {this.handleOutputLengthChange(e.target.value)}}
                                        />
                                    </div>
                                </div>
                            </div>
                            <div className="col-md-8">
                                {this.state.generated_sample.sample}
                            </div>
                        </div>
                        <div className="row">
                            <button className="btn btn-outline-dark" type="button" onClick={() => {this.generateText('zizek')}}>ZIZEK</button>
                            <button className="btn btn-outline-dark" type="button" onClick={() => {this.generateText('shakespeare')}}>SHAKESPEARE</button>
                            <button className="btn btn-outline-dark" type="button" onClick={() => {this.generateText('graham')}}>GRAHAM</button>
                        </div>
                    </div>
                </main>
                <footer></footer>
            </div> 
        );
    }
}
