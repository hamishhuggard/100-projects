<?php
function timesTable($upto) {
    echo '<table>';
    echo '<tr style="background-color: #f0f0f0">';
    echo '<th>*</th>';
    for ($col=1; $col<=$upto; $col++) {
        echo "<th>$col</th>";
    }
    echo '</tr>';
    for ($row=1; $row<=$upto; $row++) {
        echo '<tr>';
        echo "<td style='background-color: #f0f0f0'>$row</td>";
        for ($col=1; $col<=$upto; $col++) {
            $result = $row*$col;
            echo "<td>$result</td>";
        }
        echo '</tr>';
    }
}
?>

<?php timesTable(12) ?>
