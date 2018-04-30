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
            temperature: 0.60,
            n_chars_generate: 500,
            loading: false,
            generated_sample: null
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
        if (!this.state.generated_sample) {
            return ''
        }
        let loaded_sample = this.state.generated_sample;
        let display_name = null;
        switch (loaded_sample.dataset) {
            case 'zizek':
                display_name = 'Slavoj Žižek';
                break;
            case 'shakespeare':
                display_name = 'William Shakespeare';
                break;
            case 'graham':
                display_name = 'Paul Graham';
                break;
            default:
                display_name = '';
        }
        return (
            <blockquote className="blockquote">
                <div className="display-linebreak">
                    <p className="mb-0">{loaded_sample.sample}</p>
                </div>
                <footer className="blockquote-footer">{display_name + 'bot on '}<cite title="Source Title">{loaded_sample.primer.trim()}</cite></footer>
            </blockquote>
        )
    }

    render() {
        return (
            <div>
                <header></header>
                <main role="main">
                    <div className="container">
                        <h1 className="display-3 my-5">
                            ŽIŽEK GENERATOR
                        </h1>
                        <div className="row">
                            <div className="col-md-2">
                                 
                                SEED TEXT
                                <div className="row">
                                    <div className="col-12">
                                        <TextField
                                            id="field-primer:"
                                            type="text"
                                            fullWidth={true}
                                            style={{marginBottom: 40}}
                                            value={this.state.primer}
                                            onChange={(e) => {this.handleSeedChange(e.target.value)}}
                                        />
                                    </div>
                                </div>
                                
                                TEMPERATURE
                                <div className="row">
                                    <div className="col-7">
                                        <Slider
                                            min={.01}
                                            max={1.00}
                                            step={.01}
                                            onChange={(e, v) => {this.handleTemperatureChange(v)}}
                                            value={this.state.temperature}
                                        />
                                    </div>
                                    <div className="col-5">
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
                                
                                # CHARACTERS
                                <div className="row">
                                    <div className="col-7">
                                        <Slider
                                            min={1}
                                            max={999}
                                            step={1}
                                            onChange={(e, v) => {this.handleOutputLengthChange(v)}}
                                            value={this.state.n_chars_generate}
                                        />
                                    </div>
                                    <div className="col-5">
                                        <TextField
                                            id="field-chars"
                                            type="number"
                                            min={1}
                                            max={999}
                                            step={1}
                                            fullWidth={true}
                                            value={this.state.n_chars_generate}
                                            onChange={(e) => {this.handleOutputLengthChange(e.target.value)}}
                                        />
                                    </div>
                                </div>
                                <div className="form-row">
                                    <div className="col-4">
                                        <button className="btn btn-block btn-outline-dark" type="button" onClick={() => {this.generateText('zizek')}}>SZ</button>
                                    </div>
                                    <div className="col-4">
                                        <button className="btn btn-block btn-outline-dark" type="button" onClick={() => {this.generateText('shakespeare')}}>WS</button>
                                    </div>
                                    <div className="col-4">
                                        <button className="btn btn-block btn-outline-dark" type="button" onClick={() => {this.generateText('graham')}}>PG</button>
                                    </div>
                                </div>
                            </div>
                            <div className="col-md-5">
                                {this.renderSample()}
                            </div>
                            <div className="col-md-5">
                                {this.renderSample()}
                            </div>
                        </div>
                    </div>
                </main>
                <footer></footer>
            </div> 
        );
    }
}
