import React from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    FlatList,
    StyleSheet,
} from 'react-native';

const Connections = ({ connections, onSelectConnection }) => {
    const renderItem = ({ item }) => (
        <TouchableOpacity
            style={styles.connectionItem}
            onPress={() => onSelectConnection(item.serverUrl, item.websocketUrl)}>
            <Text style={styles.connectionText}>{item.serverUrl}</Text>
        </TouchableOpacity>
    );

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Connected Servers</Text>
            <FlatList
                data={connections}
                renderItem={renderItem}
                keyExtractor={item => item.serverUrl}
            />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    title: {
        fontSize: 24,
        margin: 20,
    },
    connectionItem: {
        padding: 15,
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
    },
    connectionText: {
        fontSize: 18,
    },
});

export default Connections;
