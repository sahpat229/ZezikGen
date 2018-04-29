import React from 'react';

export default class AppContainer extends React.Component {
    constructor(props) {
        super(props);
        this.handleSeedChange = this.handleSeedChange.bind(this);
        this.handleTemperatureChange = this.handleTemperatureChange.bind(this);
        this.handleOutputLengthChange = this.handleOutputLengthChange.bind(this);
        this.state = {
            seed: '',
            temperature: 1.0,
            output_length: 600,
            loading: false
        };
    }

    handleSeedChange(e) {
        this.setState({seed: e.target.value.toUpperCase()}, () => {return console.log(this.state)});
    }

    handleTemperatureChange() {
        this.setState({temperature: e.target.value}, () => {return console.log(this.state)});
    }

    handleOutputLengthChange() {
        this.setState({output_length: e.target.value}, () => {return console.log(this.state)});
    }

    generateText() {
            $.ajax({
                type: 'POST',
                url: '/refresh_data',
                data: {reports: JSON.stringify(this.state.loadedReports)},
                success: () => {this.retrieveData(query, this.state.loadedReports)}
});
    }

    render() {
        return (
            <div>
                <header></header>
                <main role="main">
                    <div className="container">
                            <h1 className="display-4">GENERATE IDEOLOGY</h1>
                            <form>
                              <div className="form-row">
                                <div className="col-md-10 mb-3">
                                    <label htmlFor="input-seed">SEED TEXT</label>
                                    <input type="text" id="input-seed" value={this.state.seed} onChange={this.handleSeedChange} className="form-control" placeholder="" />
                                </div>
                                <div className="col-md-1 mb-3">
                                    <label htmlFor="input-temperature">TEMP</label>
                                      <input type="number" id="input-temperature" value={this.state.temperature} onChange={this.handleTemperatureChange} className="form-control" placeholder="" />
                                </div>
                                <div className="col-md-1 mb-3">
                                    <label htmlFor="input-outputlength">LENGTH</label>
                                      <input type="number" id="input-outputlength" value={this.state.output_length} onChange={this.handleOutputLengthChange} className="form-control" placeholder="" />
                                </div>
                                    <button className="btn btn-outline-light" type="button">ZIZEK</button>
                                    <button className="btn btn-outline-light" type="button">SHAKESPEARE</button>
                                    <button className="btn btn-outline-light" type="button">GRAHAM</button>
                              </div>
                            </form>
                            <div className="input-group">

                            </div>
                    </div>
                </main>
                <footer></footer>
            </div>
        );
    }
}
