:- dynamic room/1.
:- dynamic connected/2.
:- dynamic furniture/2.
:- dynamic device/2.
:- dynamic status/2.
:- dynamic pocket/1.
:- retractall(pocket(_)).

project_description :-
    write('Project Description: Collect All Items'), nl,
    write('Your goal is to collect all the specified items from various furniture in different rooms.'), nl,
    write('Use the following commands to navigate and interact with the environment:'), nl,
    write('  - check_pocket: Check the items in your pocket.'), nl,
    write('  - move(Room1, Room2): Move from Room1 to Room2.'), nl,
    write('  - collect_item(Item, Furniture): If you in the right room and you can collect item'), nl,
    write('  - path(Room1, Room2, Path): Shows you a path from Room1 to Room2'), nl,
   	write('  - all_items_collected.: Shows you which items, you still need to collect'), nl,
    write('Additional commands, that may help you:'), nl,
    write('  - get_furniture_in_room(Room, Furniture): List of all furniture in a given room'), nl,
    write('  - get_devices_in_room(Room, Devices): List of all devices in a given room'),nl,
    write('  - check_devices_in_room(Room):Rule to check the status of all devices in a room'),nl,
    write('  - turn_on(Device)/turn_off(Device): turn on/turn_off given Device'),nl,
    write('The items you need to collect are:'), nl,
    write('    [energy_drink,gum,pen]'),nl.

i_am :-
    clause(i_am(Room), true),
    write('You are currently in the '), write(Room), write('.'), nl.

i_am :-
    write('Your current location is unknown.'), nl.

room(bedroom).
room(kitchen).
room(living_room).
room(bathroom).

item(energy_drink, refrigerator).
item(gum, coffee_table).
item(pen, sink).

connected(kitchen, living_room).
connected(living_room, kitchen).
connected(bedroom, living_room).
connected(living_room, bedroom).
connected(bathroom, bedroom).
connected(bedroom, bathroom).

furniture(living_room, couch).
furniture(living_room, coffee_table).
furniture(kitchen, dining_table).
furniture(kitchen, refrigerator).
furniture(bedroom, bed).
furniture(bathroom, sink).
furniture(bathroom, toilet).

device(living_room, tv).
device(kitchen, oven).
device(bedroom, lamp).
device(bathroom, shower).

status(tv, off).
status(oven, off).
status(lamp, on).
status(laptop,off).
status(shower, off).

enter(Room) :-
    retractall(i_am(_)),
    assertz(i_am(Room)),
    write('Entering '), write(Room), nl.

move(Room1, Room2) :-
    connected(Room1, Room2),
    enter(Room2).

turn_on(Device) :-
    status(Device, off),
    retractall(status(Device, _)),
    assertz(status(Device, on)),
    write(Device), write(' turned on.'), nl.

turn_off(Device) :-
    status(Device, on),
    retractall(status(Device, _)),
    assertz(status(Device, off)),
    write(Device), write(' turned off.'), nl.

% Recursive rule to find a path between two rooms
path(Room1, Room2, [Room1, Room2]) :-
    connected(Room1, Room2).

path(Room1, Room2, [Room1 | RestOfPath]) :-
    connected(Room1, IntermediateRoom),
    path(IntermediateRoom, Room2, RestOfPath).

% Rule to check if a device is in a specific room
device_in_room(Device, Room) :-
    device(Room, Device).

% Rule to find all devices in a room
devices_in_room(Room, Devices) :-
    findall(Device, device_in_room(Device, Room), Devices).

% Rule to check the status of all devices in a room
check_devices_in_room(Room) :-
    devices_in_room(Room, Devices),
    member(Device, Devices),
    status(Device, Status),
    write(Device), write(' is '), write(Status), write('.'), nl,
    fail.
check_devices_in_room(_).

% Rule to get a list of all furniture in a room
get_furniture_in_room(Room, Furniture) :-
    findall(F, furniture(Room, F), Furniture).

% Rule to get a list of all devices in a room
get_devices_in_room(Room, Devices) :-
    findall(Device, device(Room, Device), Devices).

add_item_to_pocket(Item) :-
    assertz(pocket(Item)),
    write('Added '), write(Item), write(' to your pocket.'), nl.

check_pocket :-
    findall(Item, pocket(Item), Items),
    write('Items in your pocket: '), write(Items), nl.

collect_item(Item, Furniture) :-
    i_am(Room),
    room(Room), 
    furniture(Room, Furniture),
    item(Item, Furniture),
    write('You found an item: '), write(Item), write(' (located in '), write(Furniture),
    write(' in the '), write(Room), write(')'), nl,
    add_item_to_pocket(Item).

collect_item(_, _) :-
    write('You are not in the appropriate room/furniture to collect this item.'), nl.

all_items_collected :-
    item(Item, Furniture),
    \+ pocket(Item),
    write('You still need to collect '), write(Item), write(' from '), write(Furniture), write('.'), nl,
    fail.
all_items_collected :-
    write('Congratulations! You have collected all the items.'), nl.

