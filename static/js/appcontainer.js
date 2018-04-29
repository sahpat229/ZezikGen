import React from 'react';

export default class AppContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    render() {
        return (
            <div>
                <header></header>
                <main role="main">
                    <div class="container-fluid">
                        <div class="row">
                            <div class="col-md-12">
                                <h1>Generate Ideology</h1>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-4">
                                <h2>shakespeare</h2>
                                <p>Donec id elit non mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus. Etiam porta sem malesuada magna mollis euismod. Donec sed odio dui. </p>
                                <p><a class="btn btn-secondary" href="#" role="button">View details &raquo;</a></p>
                            </div>
                            <div class="col-md-4">
                                <h2>zizek</h2>
                                <p>Donec id elit non mi porta gravida at eget metus. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus. Etiam porta sem malesuada magna mollis euismod. Donec sed odio dui. </p>
                                <p><a class="btn btn-secondary" href="#" role="button">View details &raquo;</a></p>
                            </div>
                            <div class="col-md-4">
                                <h2>xd</h2>
                                <p>Donec sed odio dui. Cras justo odio, dapibus ac facilisis in, egestas eget quam. Vestibulum id ligula porta felis euismod semper. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus.</p>
                                <p><a class="btn btn-secondary" href="#" role="button">View details &raquo;</a></p>
                            </div>
                        </div>
                    </div>
                </main>
                <footer></footer>
            </div>
        );
    }
}
