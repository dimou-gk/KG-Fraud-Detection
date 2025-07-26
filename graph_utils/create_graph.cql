:auto LOAD CSV WITH HEADERS FROM 'path_to_csv' AS ds
call {
    with ds
    match (orig:User {name: ds.nameOrig})
    match (dest:User {name: ds.nameDest})
    match (time:Time {hour: toInteger(ds.step)})
    create (trans:Transaction {transType: ds.type, amount: toFloat(ds.amount), isFraud: toInteger(ds.isFraud)})
    create (orig)-[:MADE_A {oldbalanceOrg: toFloat(ds.oldbalanceOrg), newbalanceOrg: toFloat(ds.newbalanceOrig)}]->(trans)
    create (trans)-[:WITH_ {oldbalanceDest: toFloat(ds.oldbalanceDest), newbalanceDest: toFloat(ds.newbalanceDest)}]->(dest)
    create (trans)-[:AT]->(time)
}in transactions of 10000 rows