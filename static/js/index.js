import React from 'react';
import ReactDOM from 'react-dom';

import darkBaseTheme from 'material-ui/styles/baseThemes/darkBaseTheme';
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import getMuiTheme from 'material-ui/styles/getMuiTheme';

import AppContainer from './appcontainer.js'

ReactDOM.render(
    <MuiThemeProvider>
        <AppContainer />
    </MuiThemeProvider>,
    document.getElementById('app-container')
)
