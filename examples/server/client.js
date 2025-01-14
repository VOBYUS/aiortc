// get DOM elements
// var dataChannelLog = document.getElementById('data-channel'),
//     iceConnectionLog = document.getElementById('ice-connection-state'),
//     iceGatheringLog = document.getElementById('ice-gathering-state'),
//     signalingLog = document.getElementById('signaling-state');

// peer connection
var pc = null;
// data channel
var dc = null, dcInterval = null;

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];

    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    // pc.addEventListener('icegatheringstatechange', function() {
    //     iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
    // }, false);

    // pc.addEventListener('iceconnectionstatechange', function()  
    //     iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    // }, false);
    // iceConnectionLog.textContent = pc.iceConnectionState;

    // pc.addEventListener('signalingstatechange', function() {
    //     signalingLog.textContent += ' -> ' + pc.signalingState;
    // }, false);
    // signalingLog.textContent = pc.signalingState;

    // connect audio / video

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        var codec;

        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: 'none'
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}
function display(message) {
    var data = JSON.parse(message);
    document.getElementById('basicstats').innerHTML = data.blinkCount;
    var graph = document.getElementById('graph');
    var size = graph.getBoundingClientRect();
    var height = size.height;
    if(data.drowsy_level.startsWith("waiting")){return;}
    console.log(data.drowsy_level);
    drowsyLevel = JSON.parse(data.drowsy_level)[0];
    drowsyLevel = drowsyLevel - 5;
    var bar = document.getElementById('bar');
    bar.innerHTML = '';
    var ratio = height/10;
    var barheight = drowsyLevel*ratio;
    bar.style.width = size.width + "px";
    bar.style.height = barheight + 'px';
    bar.style.backgroundColor = drowsyLevel > 4.85 ? 'red' : 'teal';
    bar.style.bottom = 0;
    bar.style.left = 0;
    bar.style.position = 'absolute';
    graph.style.position = 'relative';
    var alert = document.getElementById('alert');
    if (drowsyLevel >= 4.9) {
        alert.innerHTML = '😪'
    }
    if (drowsyLevel < 4.9 && drowsyLevel >= 4.85) {
        alert.innerHTML = '😴'
    }
    if (drowsyLevel < 4.85 && drowsyLevel >= 4.8) {
        alert.innerHTML = '🥱'
    }
    if (drowsyLevel < 4.8 && drowsyLevel >= 0) {
        alert.innerHTML = '👀'
    }

    /*
    1. get size of graph div
    2. get drowsy detection data from message
    3. create a div
    4. size the div based on drowsy detection level
    5. put the div inside the graph div
    */
}

function start() {
    document.getElementById('start').style.display = 'none';
    pc = createPeerConnection();

    var time_start = null;

    function current_stamp() {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    }
        var parameters = {"ordered": true}

        dc = pc.createDataChannel('chat', parameters);
        dc.onclose = function() {
            clearInterval(dcInterval);
            // dataChannelLog.textContent += '- close\n';
        };
        dc.onopen = function() {
            // dataChannelLog.textContent += '- open\n';
            dcInterval = setInterval(function() {
                var message = 'ping ' + current_stamp();
                // dataChannelLog.textContent += '> ' + message + '\n';
                dc.send(message);
            }, 1000);
        };
        dc.onmessage = function(evt) {
            // dataChannelLog.textContent += '< ' + evt.data + '\n'    
            display(evt.data)
           
        };

    var constraints = {
        audio: false,
        video: false
    };

        var resolution = "1280x960"
        if (resolution) {
            resolution = resolution.split('x');
            constraints.video = {
                width: parseInt(resolution[0], 0),
                height: parseInt(resolution[1], 0)
            };
        } else {
            constraints.video = true;
        }
    

    if (constraints.audio || constraints.video) {
        if (constraints.video) {
            // document.getElementById('media').style.display = 'block';
        }
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            stream.getTracks().forEach(function(track) {
                pc.addTrack(track, stream);
            });
            return negotiate();
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate();
    }

    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';

    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')
    
    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }

            match = lines[i].match(rtxRegex);
            if (match && allowed.includes(parseInt(match[2]))) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}
